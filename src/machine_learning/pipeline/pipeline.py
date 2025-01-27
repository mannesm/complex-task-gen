from loguru import logger

from models.tokenizer import Tokenizer
from .constants import BASIC_MATH_PROMPT_NO_REASONING
from .util import extract_numeric_value, reference_patterns
from .research_datasets.gsm8k_dataset import GSM8KDataset
from models.model_factory import get_model
from evaluation.pass_k import PassAtK
import sys

logger.add(sys.stderr, level="INFO")

MAX_SAMPLE_SIZE = 5

def load_dataset(dataset_name: str):
    logger.info("Loading the GSM8K dataset (test split for evaluation)")
    if dataset_name == 'gsm8k':
        return GSM8KDataset(split="test").dataset
    else:
        logger.error(f"Unknown dataset name: {dataset_name}")
        logger.error("Defaulting to gsm8k")
        return GSM8KDataset(split="test").dataset

def initialize_model(model_name: str):
    logger.info("Initializing the model via factory")
    return get_model(model_name=model_name)


def process_sample(model, sample, extracted_predictions, extracted_references):
    question = sample.get("question")
    reference_answer = sample.get("answer")

    logger.info(f"Generating prediction for question: {question}")
    final_prompt = BASIC_MATH_PROMPT_NO_REASONING.format(question=question)
    logger.debug(f"Final prompt: {final_prompt}")
    prediction = model.generate_response(final_prompt)
    logger.info(f"Prediction: {prediction}, {reference_answer}")

    extracted_references.append(extract_numeric_value(reference_answer, reference_patterns))
    # extracted_predictions.append(extract_numeric_value(prediction, reference_patterns))
    logger.info(f"Extracted predictions: {extracted_predictions}, Extracted Reference: {extracted_references}")
    return prediction, reference_answer


def run_pipeline(model_name: str = "qwen_camel", dataset_name: str = "gsm8k"):
    dataset = load_dataset(dataset_name)
    model = initialize_model(model_name)
    tokenizer_model = Tokenizer(model_name)
    predictions = []
    references = []
    extracted_predictions = []
    extracted_references = []
    log_probability_list= []
    sum_log_probabilities_list = []

    for i, question_answer in enumerate(dataset):
        if i >= MAX_SAMPLE_SIZE:
            break
        logger.info(f"Processing sample {i} out of {len(dataset)}, with a max of {MAX_SAMPLE_SIZE} samples")
        prediction, reference_answer = process_sample(model, question_answer, extracted_predictions, extracted_references)

        predictions.append(prediction)
        references.append(reference_answer)

        log_probability, sum_log_probabilities = tokenizer_model.calculate_log_probability(input_text=question_answer.get("question"), output_text= prediction)
        log_probability_list.append(log_probability)
        sum_log_probabilities_list.append(sum_log_probabilities)


    logger.info("Computing the Pass@1 metric")
    pass_k_metric = PassAtK(k=1)
    pass_k_score = pass_k_metric.compute(predictions, references)
    logger.info(f"Pass@1 for Qwen (Camel) on GSM8K: {pass_k_score:.3f}")