import pandas as pd
import numpy as np
from loguru import logger

from models.base_models.constants import MODEL_NAMES
from pipeline.constants import BASIC_MATH_PROMPT
from models.base_models.tokenizer import Tokenizer


# Adjust logging level as needed
logger.level("DEBUG")


    
def generate_log_probabilities(input_dataset: pd.DataFrame, max_sample_size=10,model_name:str = MODEL_NAMES["QWEN_2_15B"],):
    """
    input_dataset: pd.DataFrame or list-like structure of {question, answer} dicts
    max_sample_size: number of samples to process

    Returns:
        pd.DataFrame with columns:
        "question", "answer", "formatted_prompt", "tokens", "log_p", "log_p_sum",
        "num_tokens", "avg_log_p", "perplexity"
    """

    model_results = []
    tokenizer_model = Tokenizer(model_name)
    # Iterate over your input

    for i, row in input_dataset.iterrows():
        if i >= max_sample_size:
            break

        question = row.get("problem")
        answer = row.get("solution")
        question_id = row.get("id")
        # Format the prompt
        formatted_prompt_qa_pair = BASIC_MATH_PROMPT.format(question=question, answer=answer)
        formatted_prompt_question = BASIC_MATH_PROMPT.format(question=question, answer="")

        logger.info(f"Processing sample {i} out of {len(input_dataset)}, " f"with a max of {max_sample_size} samples")
        logger.debug(f"Final prompt: {formatted_prompt_qa_pair}")

        tokens, log_p, log_p_sum = tokenizer_model.calculate_log_probability(
            text=formatted_prompt_qa_pair, prompt=formatted_prompt_question
        )

        model_results.append((question_id, question, answer, formatted_prompt_qa_pair, tokens, log_p, log_p_sum))
    df = pd.DataFrame(model_results, columns=["question_id","question", "answer", "formatted_prompt", "tokens", "log_p", "log_p_sum"])

    df["num_tokens"] = df["tokens"].apply(len)
    df["avg_log_p"] = df["log_p_sum"] / df["num_tokens"]
    df["perplexity"] = np.exp(-df["avg_log_p"])

    return df
