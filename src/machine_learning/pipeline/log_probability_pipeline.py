import pandas as pd
import numpy as np
from loguru import logger
from constants import BASIC_MATH_PROMPT
from models.base_models.tokenizer import Tokenizer


# Adjust logging level as needed
logger.level("DEBUG")

def generate_log_probabilities(input_dataset: pd.DataFrame, max_sample_size=5):
    """
    input_dataset: pd.DataFrame or list-like structure of {question, answer} dicts
    max_sample_size: number of samples to process

    Returns:
        pd.DataFrame with columns:
        "question", "answer", "formatted_prompt", "tokens", "log_p", "log_p_sum",
        "num_tokens", "avg_log_p", "perplexity"
    """

    model_results = []

    # Iterate over your input
    tokenizer_model = Tokenizer()

    for i, row in input_dataset.iterrows():
        if i >= max_sample_size:
            break

        question = row.get("question")
        answer = row.get("answer")

        # Format the prompt
        formatted_prompt = BASIC_MATH_PROMPT.format(question=question, answer=answer)

        logger.info(f"Processing sample {i} out of {len(input_dataset)}, " f"with a max of {max_sample_size} samples")
        logger.debug(f"Final prompt: {formatted_prompt}")

        # Calculate log probabilities
        tokens, log_p, log_p_sum = tokenizer_model.calculate_log_probability(formatted_prompt)

        # Store raw results
        model_results.append((question, answer, formatted_prompt, tokens, log_p, log_p_sum))

    # Create DataFrame
    df = pd.DataFrame(model_results, columns=["question", "answer", "formatted_prompt", "tokens", "log_p", "log_p_sum"])

    # 1. Compute number of tokens
    df["num_tokens"] = df["tokens"].apply(len)

    # 2. Compute average log probability (normalized by length)
    df["avg_log_p"] = df["log_p_sum"] / df["num_tokens"]

    # 3. Compute perplexity
    #    Perplexity = exp( - average log probability )
    df["perplexity"] = np.exp(-df["avg_log_p"])
    return df