import pandas as pd
from loguru import logger

from models.base_models.base_camel import BaseModel
from models.base_models.constants import MODEL_NAMES
from models.base_models.tokenizer import Tokenizer
from pipeline.prompts import BASIC_MATH_PROMPT
from research_datasets.math_dataset import MathDataset
from util import extract_numeric_value
# from constants import prediction_patterns
from research_datasets.gsm8k_dataset import GSM8KDataset
from models.base_models.model_factory import get_model
from machine_learning.evaluation.pass_k import PassAtKCalculator
import sys

logger.add(sys.stderr, level="INFO")

MAX_SAMPLE_SIZE = 5


def load_dataset(dataset_name: str, split: str= "test"):
    logger.info("Loading the GSM8K dataset (test split for evaluation)")
    if dataset_name == "gsm8k":
        if split == 'train':
            return GSM8KDataset(split="train").dataset
        return GSM8KDataset(split="test").dataset
    if dataset_name == "math":
        if split == "train":
            return MathDataset(split="train").dataset

    else:
        logger.error(f"Unknown dataset name: {dataset_name}")
        logger.error("Defaulting to gsm8k")
        raise NotImplementedError("Unknown dataset name")


def initialize_model(model_name: str):
    logger.info("Initializing the model via factory")
    return get_model(model_name=model_name)


def format_prompt(question: str):
    return BASIC_MATH_PROMPT.format(question=question)


def process_sample(model: BaseModel, question: str) -> tuple[str, str]:
    logger.info(f"Generating prediction for question: {question}")
    formatted_prompt = format_prompt(question)
    logger.debug(f"Final prompt: {formatted_prompt}")
    prediction = model.generate_response(formatted_prompt)
    logger.info(f"Generated Answer: {prediction}")
    return formatted_prompt, prediction



def run_pipeline(model_name: str = "qwen_camel", dataset_name: str = "gsm8k"):
    dataset = load_dataset(dataset_name)
    model = initialize_model(model_name)
    tokenizer = Tokenizer(model_name)
    results = []

    for i, dataset_question_answer in enumerate(dataset):
        if i >= MAX_SAMPLE_SIZE:
            break
        logger.info(f"Processing sample {i} out of {len(dataset)}, with a max of {MAX_SAMPLE_SIZE} samples")

        question = dataset_question_answer.get("question")
        reference_answer = dataset_question_answer.get("answer")
        extracted_numeric_reference_answer = extract_numeric_value(reference_answer, prediction_patterns)
        formatted_prompt, generated_answer = process_sample(model, question)
        extracted_numeric_predicted_answer = extract_numeric_value(generated_answer, prediction_patterns)

        tokenized_input, log_probabilities, log_probability_sum = tokenizer.calculate_log_probability(text=(formatted_prompt+reference_answer))
        tokenized_input, prompt_answer_log_p, prompt_answer_log_p_sum = tokenizer.calculate_log_probability(
            text=(formatted_prompt + reference_answer)
        )

        answer_log_p, answer_log_p_sum = tokenizer.calculate_log_probability(text=reference_answer)

        results.append(
            {
                "question": dataset_question_answer.get("question"),
                "prediction": generated_answer,
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
    pass_k_metric = PassAtKCalculator(k=1)
    # pass_k_score = (df_results["prediction"].tolist(), df_results["reference_answer"].tolist())
    # logger.info(f"Pass@1 for Qwen (Camel) on GSM8K: {pass_k_score:.3f}")

    return df_results

QWEN_SMALL = MODEL_NAMES.get("QWEN_2_MATH_7B")
MODEL_NAMES
run_pipeline(model_name=QWEN_SMALL, dataset_name="gsm8k")