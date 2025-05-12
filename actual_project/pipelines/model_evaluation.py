import re
import logging

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import os

from pipelines.gsm_evaluation_dataset_creation import create_gsm_evaluation_datasets

# Setup logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
file_name = "evaluation_results_qwen_finetune_base_gsm8k.csv"

RESULTS_FILE = f"/home/mmokkenstorm/{file_name}"
SAVE_EVERY = 5  # Save after every N examples

SYMBOLIC_HUGGINFACE_NAME = "apple/GSM-Symbolic" # todo: also implement the medium and hard questions with the same index / augmented question answer pair.\
MATH_HUGGINGFACE_NAME = "HuggingFaceH4/MATH-500"
GSM8K_HUGGINGFACE_NAME = "openai/gsm8k"

create_gsm_evaluation_datasets(to_df=False)

def extract_answer_from_gsm_dataset(example):
    answer = example["answer"]
    match = re.search(r'####\s*(\d+)', answer)
    if match:
        return match.group(1)
    return None

def rename_math_columns(example):
    example["question"] = example.pop("problem")  # <-- Fix
    example["actual_extracted_answer"] = example.pop("answer")
    example["answer"] = example.pop("solution")
    return example

def evaluate_math_model(model_name, system_prompt, split="test", max_examples=10):
    logging.info(f"Loading model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # tokenizer.chat_template = (
        #     "{% for message in messages %}"
        #     "{% if message['role'] == 'system' %}"
        #     "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        #     "{% elif message['role'] == 'user' %}"
        #     "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        #     "{% elif message['role'] == 'assistant' %}"
        #     "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        #     "{% endif %}"
        #     "{% endfor %}"
        #     "<|im_start|>assistant\n"
        # )  #TODO: Remove add generation prompt
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        logging.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


    results = []
    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_csv(RESULTS_FILE, sep="|")
        already_done = set(zip(df_existing["dataset"], df_existing["question"]))
        logging.info(f"Found {len(already_done)} previously processed examples.")
    else:
        df_existing = pd.DataFrame()
        already_done = set()

    for dataset_name, sub_name in datasets:
        logging.info(f"Loading dataset: {dataset_name}")
        load_process_dataset(dataset_name, sub_name=sub_name, split=split)
        for i, example in enumerate(dataset):
            if (dataset_name, example["question"]) in already_done:
                logging.info(f"Skipping already processed example {i + 1}")
                continue
            if i >= max_examples:
                break
            try:
                question = example["question"]
                real_answer = example["actual_extracted_answer"]

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  #TODO: Remove add generation prompt #
                response = generator(prompt, max_new_tokens=1024)[0]["generated_text"]
                new_text = response[len(prompt) :].strip()

                patterns = [
                    r"\\boxed\{(.+?)\}",
                    r"boxed\{(.+?)\}",
                    r"\[?box(?:ed)?\]?\s*\{?([^\}\n]+)\}?",
                ]
                logging.info(f"Generated text: {new_text}")
                generated_answer = None
                for pattern in patterns:
                    match = re.search(pattern, new_text)
                    if match:
                        generated_answer = match.group(1).strip()
                        logging.info(f"Generated answer: {generated_answer}")
                        break

                is_correct = (generated_answer == real_answer)

                results.append({
                    "question": question,
                    "answer": real_answer,
                    "generated_answer": generated_answer,
                    "full_generated_answer": new_text,
                    "dataset": dataset_name,
                    "correct": is_correct,
                })
                if len(results) % SAVE_EVERY == 0:
                    pd.DataFrame(results).to_csv(RESULTS_FILE, mode='a', index=False, sep="|", header=not os.path.exists(RESULTS_FILE))
                    results.clear()

                logging.info(f"[{dataset_name}] Example {i+1}/{max_examples} processed. Correct: {is_correct}")

            except Exception as e:
                logging.error(f"Error processing example {i} in {dataset_name}: {e}")
                continue

    df = pd.DataFrame(results)
    if results:
        pd.DataFrame(results).to_csv(
            RESULTS_FILE,
            mode="a",
            index=False,
            sep="|",
            header=not os.path.exists(RESULTS_FILE),
        )

    logging.info("Evaluation completed.")
    return df

system_prompt = """You are a math reasoning assistant.
Your job is to solve math problems step by step, showing your reasoning clearly and logically.

Instructions:
1. Break the problem into smaller steps and explain each one.
2. Justify each step, explaining why it is valid.
3. Highlight any assumptions or edge cases that may affect the solution.
4. Conclude with the final result using the format:

Final Answer:
\\boxed{your final answer here}

Only include one boxed expression at the end of your response.
"""

#TODO: Tokenize it and then detokenize to see if I apply chat template correctly



# model_name = "/home/mmokkenstorm/sync/qwen_models/Qwen2.5-Math-7B/final"  # Replace with your model
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
evaluation_results = evaluate_math_model(model_name, system_prompt, max_examples=300)
if evaluation_results is not None:
    os.chdir("/home/mmokkenstorm")

    evaluation_results.to_csv(file_name, index=False, sep="|")
    logging.info(f"Saved results to {file_name}")



import html

def clean_extracted_answer(answer):
    """
    Cleans up LaTeX or symbolic answers like '20\\%' or '\\frac{1}{2}'.
    """
    if not answer:
        return None
    # Decode LaTeX-escaped % and others
    answer = html.unescape(answer)
    answer = answer.replace("\\%", "%")
    answer = answer.replace("\\,", "")  # remove LaTeX spacing
    answer = answer.replace("$", "")    # remove math mode markers
    answer = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", answer)  # crude unboxing \frac{}{}
    return answer.strip()