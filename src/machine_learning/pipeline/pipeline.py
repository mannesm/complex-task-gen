from loguru import logger
from .research_datasets.gsm8k_dataset import GSM8KDataset
from .models.model_factory import get_model
from .evaluation.pass_k import PassAtK, GSM8KPassAtK

def run_pipeline():
    logger.info("Loading the GSM8K dataset (test split for evaluation)")
    # 1. Load the GSM8K dataset (test split for evaluation)
    logger.debug('Loading the GSM8K dataset (test split for evaluation)')
    dataset = GSM8KDataset(split="test").dataset

    logger.info("Initializing the model via factory")
    # 2. Initialize the model via factory
    #    If you prefer CPU or MPS on a Mac, pass device="cpu" or "mps"
    model = get_model(model_name="qwen_camel", device="mps")

    logger.info("Starting inference on the dataset")
    # 3. Inference: Loop through a subset of the dataset, collect predictions & references
    predictions = []
    references = []

    # Use a small subset to test quickly, e.g. dataset[:5]
    i = 0
    for sample in dataset:
        question = sample.get("question")
        reference_answer = sample.get("answer")

        logger.debug(f"Generating prediction for question: {question}")
        # Generate prediction from the model
        prediction = model.generate_solution(question)
        predictions.append(prediction)
        references.append(reference_answer)
        i += 1
        if i >= 2:
            break
    logger.info("Computing the Pass@1 metric")
    # 4. Compute the Pass@1 metric
    if dataset.info.dataset_name == 'gsm8k':
        pass_k_metric = GSM8KPassAtK(k=1)
    else:
        pass_k_metric = PassAtK(k=1)
    pass_k_score = pass_k_metric.compute(predictions, references)

    logger.info(f"Pass@1 for Qwen (Camel) on GSM8K: {pass_k_score:.3f}")