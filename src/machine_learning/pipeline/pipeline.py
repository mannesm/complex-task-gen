from loguru import logger
from .research_datasets.gsm8k_dataset import GSM8KDataset
from .models.model_factory import get_model
from evaluation.pass_k import PassAtK, GSM8KPassAtK
import sys


logger.add(sys.stderr, level="INFO")



def run_pipeline(model_name: str = "qwen_camel", dataset_name: str = "gsm8k"):
    logger.info("Loading the GSM8K dataset (test split for evaluation)")
    if dataset_name == 'gsm8k':
        dataset = GSM8KDataset(split="test").dataset
    else:
        logger.error(f"Unknown dataset name: {dataset_name}")
        logger.error("Defaulting to gsm8k")
        dataset = GSM8KDataset(split="test").dataset

    logger.info("Initializing the model via factory")
    model = get_model(model_name=model_name)

    logger.info("Starting inference on the dataset")
    predictions = []
    references = []

    i = 0
    max_sample_size = 20
    for sample in dataset:
        logger.info(f"Processing sample {i} out of {len(dataset)}, with a max of {max_sample_size} samples")
        question = sample.get("question")
        reference_answer = sample.get("answer")

        logger.info(f"Generating prediction for question: {question}")
        prediction = model.generate_solution(question)
        logger.info(f"Prediction: {prediction}, {reference_answer}")
        predictions.append(prediction)
        references.append(reference_answer)
        i += 1
        if i >= max_sample_size:
            break
    logger.info("Computing the Pass@1 metric")
    # 4. Compute the Pass@1 metric
    if dataset.info.dataset_name == 'gsm8k':
        pass_k_metric = GSM8KPassAtK(k=1)
    else:
        pass_k_metric = PassAtK(k=1)
    pass_k_score = pass_k_metric.compute(predictions, references)

    logger.info(f"Pass@1 for Qwen (Camel) on GSM8K: {pass_k_score:.3f}")