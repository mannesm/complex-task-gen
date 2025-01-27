import torch
from torch import log_softmax


def compute_log_probability(tokens, token_ids ):
    """
    Computes the log probability of a given solution for a task.

    Args:
        prompt (str): The task prompt.
        solution (str): The proposed solution.

    Returns:
        float: Log probability of the solution.
    """

    log_probs = log_softmax(logits, dim=-1)

    # Tokenize solution separately to identify its tokens
    solution_ids = self.model.tokenizer(solution, return_tensors="pt")["input_ids"].squeeze()

    # Extract log-probs for solution tokens (align with the end of the input sequence)
    solution_start = input_ids.size(1) - solution_ids.size(0)  # Start index for solution tokens
    solution_log_probs = log_probs[0, solution_start:, solution_ids].sum()

    return solution_log_probs.item()
