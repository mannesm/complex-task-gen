import pandas as pd

from dotenv import load_dotenv
import os

from constants import PERPLEXITY_THRESHOLD, BASIC_MATH_REASONING_PROMPT
from pipeline.log_probability_pipeline import generate_log_probabilities
from models.base_models.qwen_2_math_7b import QWen2Math7BModel

from util import extract_numeric_value, prediction_patterns


os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.getenv("ENV_FILE", "./Users/mannes/thesis/.env")
load_dotenv("/Users/mannes/thesis/.env")

# 
SAMPLE_SIZE = 1000
MODEL_TEMPERATURE = 0.8
SAMPLES_PER_QUESTION = 5
SAMPLES_PER_QUESTION = range(SAMPLES_PER_QUESTION)


if __name__ == "__main__":
    from loguru import logger

    logger.info("Starting the pipeline")
    math_dataset = pd.read_json(
        "/Users/mannes/thesis/complex_task_gen/src/machine_learning/research_datasets/downloaded_gsm8k_10.json"
    )

    log_probability_df = generate_log_probabilities(input_dataset=math_dataset, max_sample_size=SAMPLE_SIZE)

    log_probability_df.to_csv("../output/log_probabilities.csv", index=False)

    difficulty_threshold = PERPLEXITY_THRESHOLD

    df = log_probability_df
    logger.info(f"Number of hard questions: {len(df)}")

    qwen_base_math = QWen2Math7BModel(temperature=MODEL_TEMPERATURE)

    results = []
    for index, row in df.iterrows():
        logger.info(f"Hard question: {row['question']}, Answer: {row['answer']}")
        for i in SAMPLES_PER_QUESTION:
            MATH_QUESTION_SOLVER = BASIC_MATH_REASONING_PROMPT.format(question=row["question"])
            generated_solution = qwen_base_math.generate_response(MATH_QUESTION_SOLVER)

            extracted_generated_solution = extract_numeric_value(generated_solution, prediction_patterns)
            extracted_actual_solution = extract_numeric_value(row["answer"], prediction_patterns)
            if extracted_generated_solution == extracted_actual_solution:
                logger.info(f"Correctly generated solution: {extracted_generated_solution}")
                correctly_generated_solution_bool = True
            else:
                logger.info(f"Incorrectly generated solution: {extracted_generated_solution}")
                correctly_generated_solution_bool = False

            logger.info(
                f"""Question: {row['question']},
                \n Generated Answer: {extracted_generated_solution},
                \n Actual Answer: {extracted_actual_solution}"""
            )

            results.append(
                {
                    "question_id": index,
                    "question": row["question"],
                    "sample_number": i,
                    "actual_answer": extracted_actual_solution,
                    "generated_answer": extracted_generated_solution,
                    "correct_prediction": correctly_generated_solution_bool,
                    "actual_full_answer": row["answer"],
                    "generated_full_answer": generated_solution,
                }
            )

        logger.info(f"Finished generating {len(SAMPLES_PER_QUESTION)} answers for this question")

    results_df = pd.DataFrame(results)
    results_df.set_index(["question_id", "sample_number"], inplace=True)


    logger.info("Pipeline finished")
    logger.info("Saving output to CSV")
    results_df.reset_index().to_csv("../output/evaluate_pass_k_log_p.csv", index=False)

    logger.info("Output saved")
