import re
import logging
import sys

from util import extract_generated_question_answer

sys.path.insert(0, "/home/mmokkenstorm/sync")


import os

from actual_project.pipelines.gsm_evaluation_dataset_creation import create_gsm_evaluation_datasets
from actual_project.models.finetuned_model import load_model
# Setup logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
file_name = "evaluation_results_qwen_finetune_base_gsm8k.csv"

RESULTS_FILE = f"/home/mmokkenstorm/{file_name}"
SAVE_EVERY = 5

SYMBOLIC_HUGGINFACE_NAME = "apple/GSM-Symbolic" # todo: also implement the medium and hard questions with the same index / augmented question answer pair.\
MATH_HUGGINGFACE_NAME = "HuggingFaceH4/MATH-500"
GSM8K_HUGGINGFACE_NAME = "openai/gsm8k"
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

gsm8k, gsm_easy, gsm_medium, gsm_hard = create_gsm_evaluation_datasets(to_df=True)

all_datasets = [
    gsm8k, gsm_easy, gsm_medium, gsm_hard
]
gsm_answer_pattern = r'####\s*(\d+)'

def extract_answer_from_gsm_dataset(example, regex_patterns: list = [r'####\s*(\d+)']):
    """
    Extracts an answer from the example using a list of regex patterns.

    :param example: A dictionary containing the "answer" key.
    :param regex_patterns: A list of regex patterns to match against the answer.
    :return: The first matched group or None if no match is found.
    """
    answer = example["answer"]
    for pattern in regex_patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(1)
    return None

def rename_math_columns(example):
    example["question"] = example.pop("problem")  # <-- Fix
    example["actual_extracted_answer"] = example.pop("answer")
    example["answer"] = example.pop("solution")
    return example


model, tokenizer, device = load_model('/gpfs/home6/mmokkenstorm/sync/qwen_models/finetuned_models/math_instruct/lora')

for question in gsm8k["question"]:
    print(question)
    print(tokenizer.tokenize(question))
    print(tokenizer.decode(tokenizer.tokenize(question)))
    print(tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True))
    print(tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True).split(")\n"))
    break

def generate_answer(question: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**input_ids, max_new_tokens=1024)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt part from the output if necessary
    return decoded.split("<|im_start|>assistant")[-1].strip()

from tqdm import tqdm
tqdm.pandas()  # Optional: Progress bar
for dataset in all_datasets:
    dataset["generated_answer"] = dataset["question"].progress_apply(generate_answer)
    dataset['extracted_actual_answer'] = dataset.progress_apply(extract_answer_from_gsm_dataset,  axis=1)
    dataset['extracted_generated_answer'] = dataset.progress_apply(, axis=1)
    gsm_easy.to_csv("/home/mmokkenstorm/model_outputs/gsm_easy.csv", index=False, sep="|")
#

#
# gsm8k.to_csv("/home/mmokkenstorm/model_outputs/gsm8k.csv", index=False, sep="|")
# gsm_hard.to_csv("/home/mmokkenstorm/model_outputs/gsm_hard.csv", index=False, sep="|")
# gsm_medium.to_csv("/home/mmokkenstorm/model_outputs/gsm_medium.csv", index=False, sep="|")



# generate_response(model, tokenizer, device, "What is 2 + 2?")



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