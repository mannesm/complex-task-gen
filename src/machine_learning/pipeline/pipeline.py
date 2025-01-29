import pandas as pd
from loguru import logger
from models.base_models.tokenizer import Tokenizer
from .constants import BASIC_MATH_PROMPT_NO_REASONING
from .util import extract_numeric_value, reference_patterns
from research_datasets.gsm8k_dataset import GSM8KDataset
from models.base_models.model_factory import get_model
from evaluation import PassAtK
import sys

logger.add(sys.stderr, level="INFO")

MAX_SAMPLE_SIZE = 5


def load_dataset(dataset_name: str):
    logger.info("Loading the GSM8K dataset (test split for evaluation)")
    if dataset_name == "gsm8k":
        return GSM8KDataset(split="test").dataset
    else:
        logger.error(f"Unknown dataset name: {dataset_name}")
        logger.error("Defaulting to gsm8k")
        return GSM8KDataset(split="test").dataset


def initialize_model(model_name: str):
    logger.info("Initializing the model via factory")
    return get_model(model_name=model_name)


def format_prompt(question: str):
    return BASIC_MATH_PROMPT_NO_REASONING.format(question=question)


def process_sample(model, sample, extracted_references):
    question = sample.get("question")
    reference_answer = sample.get("answer")

    logger.info(f"Generating prediction for question: {question}")
    formatted_prompt = format_prompt(question)
    logger.debug(f"Final prompt: {formatted_prompt}")
    prediction = model.generate_response(formatted_prompt)
    logger.info(f"Prediction: {prediction}, {reference_answer}")

    extracted_references.append(extract_numeric_value(reference_answer, reference_patterns))
    logger.info(f"Extracted Reference: {extracted_references}")
    return formatted_prompt, prediction, reference_answer


def run_pipeline(model_name: str = "qwen_camel", dataset_name: str = "gsm8k"):
    dataset = load_dataset(dataset_name)
    model = initialize_model(model_name)
    tokenizer_model = Tokenizer(model_name)

    results = []

    for i, question_answer in enumerate(dataset):
        if i >= MAX_SAMPLE_SIZE:
            break
        logger.info(f"Processing sample {i} out of {len(dataset)}, with a max of {MAX_SAMPLE_SIZE} samples")
        formatted_prompt, prediction, reference_answer = process_sample(model, question_answer, extracted_references=[])

        tokenized_input, prompt_answer_log_p, prompt_answer_log_p_sum = tokenizer_model.calculate_log_probability(
            text=(formatted_prompt + reference_answer)
        )
        # answer_log_p, answer_log_p_sum = tokenizer_model.calculate_log_probability(text=reference_answer)

        results.append(
            {
                "question": question_answer.get("question"),
                "prediction": prediction,
                "reference_answer": reference_answer,
                "full_prompt": formatted_prompt,
                "tokenized_input": tokenized_input,
                "prompt_answer_log_p": prompt_answer_log_p,
                "prompt_answer_log_p_sum": prompt_answer_log_p_sum,
                # "answer_log_p": answer_log_p,
                # "answer_log_p_sum":  answer_log_p_sum,
            }
        )

    df_results = pd.DataFrame(results)
    logger.info("Results DataFrame created")

    logger.info("Computing the Pass@1 metric")
    pass_k_metric = PassAtK(k=1)
    pass_k_score = pass_k_metric.compute(df_results["prediction"].tolist(), df_results["reference_answer"].tolist())
    logger.info(f"Pass@1 for Qwen (Camel) on GSM8K: {pass_k_score:.3f}")

    return df_results
