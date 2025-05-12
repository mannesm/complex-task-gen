import pandas as pd

from dotenv import load_dotenv
import os

from machine_learning.models.base_models.qwen_2_15b import QWen215BModel
from constants import AUGMENTATION_PROMPT, VERIFIER_PROMPT, PERPLEXITY_THRESHOLD
from models.base_models.deepseek_r1 import DeepseekR1Model
from pipeline.log_probability_pipeline import generate_log_probabilities

from research_datasets.gsm8k_dataset import GSM8KDataset
from util import extract_generated_question_answer, check_verification_result
#
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.getenv("ENV_FILE", "./Users/mannes/thesis/.env")
load_dotenv("/Users/mannes/thesis/.env")
MODEL_NAME = "QWEN_2_15B"

SAMPLE_SIZE = 1
MODEL_TEMPERATURE = 0.8

if __name__ == "__main__":
    from loguru import logger

    logger.info("Starting the pipeline")
    math_dataset = GSM8KDataset(split="train").to_dataframe()

    math_dataset = math_dataset.head(5)
    log_probability_df = generate_log_probabilities(input_dataset=math_dataset, max_sample_size=SAMPLE_SIZE)
    difficulty_threshold = PERPLEXITY_THRESHOLD

    df = log_probability_df
    # df = log_probability_df[log_probability_df["perplexity"] > difficulty_threshold]
    logger.info(f"Number of hard questions: {len(df)}")

    qwen_base_math = QWen215BModel(temperature=MODEL_TEMPERATURE)
    augmented_data = []
    for index, row in df.iterrows():
        logger.info(f"Hard question: {row['question']}, Answer: {row['answer']}")

        AUGMENTATION_PROMPT_QA = AUGMENTATION_PROMPT.format(
            original_question=row["question"], original_answer=row["answer"]
        )
        # !TODO: What if we add multiple QA pairs into one prompt?
        logger.debug(f"Augmentation prompt: {AUGMENTATION_PROMPT_QA}")

        augmented_response = qwen_base_math.generate_response(AUGMENTATION_PROMPT_QA)
        augmented_question, augmented_answer = extract_generated_question_answer(augmented_response)
        logger.info(f"Augmented question: {augmented_question}, Answer: {augmented_answer}")

        augmented_data.append(
            {
                "question": augmented_question,
                "answer": augmented_answer,
                "original_question": row["question"],
                "original_answer": row["answer"],
            }
        )
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
            logger.info(f"verifier model judgement: {verifier_answer}, the model is correctly augmenting")
            correct_verifier_output.append(row)
            
        else:
            logger.info(f"verifier model judgement: {verifier_answer}, the model is not correctly augmenting")
            incorrect_verifier_output.append(row)

    correct_df = pd.DataFrame(correct_verifier_output)
    correct_df["correctly_augmented"] = True
    incorrect_df = pd.DataFrame(incorrect_verifier_output)
    incorrect_df["correctly_augmented"] = False

    # combined_df = pd.concat([correct_df, incorrect_df])
    # # combined_df["question"] = combined_df.apply(
    # #     lambda row: row["question"] if row["correctly_augmented"] else row["original_question"], axis=1
    # # ).drop["original_question"]
    # # combined_df["answer"] = combined_df.apply(
    # #     lambda row: row["answer"] if row["correctly_augmented"] else row["original_answer"], axis=1
    # # ).drop["original_answer"]
    # #
    # # combined_df = combined_df.drop(columns=["original_question", "original_answer"])

    log_probabilities_hard_questions = generate_log_probabilities(correct_df, max_sample_size=SAMPLE_SIZE)

    logger.info("Pipeline finished")
    logger.info("Saving output to CSV")
    # combined_df.to_csv("output/combined_df.csv", index=False)
    log_probabilities_hard_questions.to_csv(f"output/log_probabilities_hard_questions{MODEL_NAME}.csv", index=False)

    logger.info("Output saved")
