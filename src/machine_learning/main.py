import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from constants import AUGMENTATION_PROMPT
from models.base_models.deepseek_r1 import DeepseekR1Model
from pipeline.log_probability_pipeline import generate_log_probabilities
from models.base_models.qwen_2_math_7b import QWen2Math7BModel
import sys
import os

from research_datasets.gsm8k_dataset import GSM8KDataset
from util import extract_generated_question_answer, check_verification_result
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger.remove()
logger.add(sys.stderr, level="INFO")
os.getenv("ENV_FILE", "./Users/mannes/thesis/.env")
load_dotenv("/Users/mannes/thesis/.env")
if __name__ == "__main__":
    logger.info("Starting the pipeline")
    math_dataset = GSM8KDataset(split="train").to_dataframe()
    log_probability_df = generate_log_probabilities(math_dataset, max_sample_size=10)
    difficulty_threshold = 12
    hard_questions_df = log_probability_df[log_probability_df["perplexity"] > difficulty_threshold]
    logger.info(f"Number of hard questions: {len(hard_questions_df)}")
    QWEN_BASE_MODEL = QWen2Math7BModel()
    augmented_data = []
    for index, row in hard_questions_df.iterrows():
        logger.info(f"Hard question: {row['question']}, Answer: {row['answer']}")
        AUGMENTATION_PROMPT_QA = AUGMENTATION_PROMPT.format(original_question = row["question"], original_answer = row["answer"]) # TODO: What if we add multiple QA pairs into one prompt?
        logger.debug(f"Augmentation prompt: {AUGMENTATION_PROMPT_QA}")
        augmented_response = QWEN_BASE_MODEL.generate_response(AUGMENTATION_PROMPT_QA)
        augmented_question, augmented_answer = extract_generated_question_answer(augmented_response)
        logger.info(f"Augmented question: {augmented_question}, Answer: {augmented_answer}")
        augmented_data.append({"question": augmented_question, "answer": augmented_answer})
        logger.info("Augmentation finished")

    augmented_df = pd.DataFrame(augmented_data)

    verifier_model = DeepseekR1Model()
    correct_verifier_output = []
    incorrect_verifier_output = []
    for index, row in augmented_df.iterrows():
        logger.info(f"Augmented question: {row['question']}, Answer: {row['answer']}")
        verifier_model_prompt = VERIFIER_PROMPT.format(question=row["question"], proposed_answer=row["answer"])
        verifier_response = verifier_model.generate_response(verifier_model_prompt)
        verifier_answer = check_verification_result(verifier_response)
        if verifier_answer:
            logger.info("verifier model judgement: {verifier_answer}, the model is correctly augmenting")
            correct_verifier_output.append(row)

        else:
            logger.info("verifier model judgement: {verifier_answer}, the model is not correctly augmenting")
            incorrect_verifier_output.append(row)
    correct_df = pd.DataFrame(correct_verifier_output)
    # incorrect_df = pd.DataFrame(incorrect_verifier_output)
    log_probabilities_hard_questions = generate_log_probabilities(correct_df, max_sample_size=10)


    augmented_log_probability_df = generate_log_probabilities(augmented_df, max_sample_size=10)


    # pipeline_output_df = run_pipeline(model_name=MODEL_NAMES["QWEN_2_MATH_7B"], dataset_name="gsm8k")
    # pipeline_output_df.to_csv("output/pipeline_output.csv", index=False)
    logger.info("Pipeline finished")