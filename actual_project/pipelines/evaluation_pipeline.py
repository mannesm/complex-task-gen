import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from complex_task_gen.actual_project.pipelines.gsm_evaluation_dataset_creation import create_gsm_evaluation_datasets
from openai import OpenAI
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

MODEL_NAME = 'Qwen/Qwen2.5-Math-7B-Instruct'
BASE_URL = 'http://localhost:8002/v1'
MAX_TOKENS_RESPONSE = 2000

client = OpenAI(base_url=BASE_URL, api_key='EMPTY')

BASIC_MATH_REASONING_PROMPT = """You are a math reasoning assistant.
Your job is to solve the following problem step by step, showing your reasoning clearly and logically.

**Question**: {problem}

1. Break the problem into smaller steps and explain each one as you proceed.
2. Justify each step of your reasoning, explaining why it is valid.
3. Highlight any assumptions or edge cases that may affect the solution.
4. Conclude with the final result using clear and logical math expressions. Format your conclusion like this:

The total is ... = $<<...=numeric_value>>numeric_value.

Always finish with the final numeric answer boxed in LaTeX format on a separate line, like this:

\\[
\\boxed{{<numeric_answer>}}
\\]
"""


TAG_RX = {
    'code': re.compile(r'<code>\s*```python\s*(.*?)\s*```.*?</code>', re.DOTALL | re.IGNORECASE),
    'task': re.compile(r'<task>(.*?)</task>', re.DOTALL | re.IGNORECASE),
    'solution': re.compile(r'<solution>(.*?)</solution>', re.DOTALL | re.IGNORECASE),
    'answer': re.compile(
        r'(?:<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>'  # alt‑1 – tags
        r'|####\s*([+-]?\d+(?:\.\d+)?)\b)',  # alt‑2 – hashes
        re.IGNORECASE,
    ),
}


def extract_answer_robust(model_output: str) -> float | None:
    """Extracts the numeric answer from model output using multiple regex patterns.
    Handles various formats including boxed answers and different number formats.

    Args:
        model_output: The text output from the model

    Returns:
        The extracted numeric value as a float, or None if no answer is found
    """
    # List of regex patterns to try in order
    patterns = [
        r'\\boxed\{([0-9\.,\-]+)\}',  # \boxed{3}
        r'\\\(\s*\\boxed\{([0-9\.,\-]+)\}\s*\\\)',  # \(\boxed{3}\)
        r'\\\[\s*\\boxed\{([0-9\.,\-]+)\}\s*\\\]',  # \[\boxed{140}\]
        r'####\s*([0-9\.,\-]+)',  # #### 140
        r'Answer:\s*([0-9\.,\-]+)',  # Answer: 140
        r'answer\s*(?:is|:)?\s*([0-9\.,\-]+)',  # answer is 140 or answer: 140
        r'=\s*([0-9\.,\-]+)\s*$',  # = 140 at end of line
        r'(\d+(?:\.\d+)?)\s*(?:is the answer|is our answer)',  # 140 is the answer
    ]

    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            try:
                # Convert to float, handling commas
                answer_text = match.group(1).replace(',', '')
                answer_value = float(answer_text)
                logging.info(f'Extracted answer: {answer_value} using pattern: {pattern}')
                return answer_value
            except (ValueError, IndexError) as e:
                logging.debug(f'Match found with pattern {pattern} but conversion failed: {e}')
                continue

    # Last resort: look for any boxed expression
    boxed_match = re.search(r'\\boxed\{([^{}]+)}', model_output)
    if boxed_match:
        try:
            # Try to evaluate simple expressions or convert to float
            answer_text = boxed_match.group(1).replace(',', '')
            # Remove LaTeX formatting like \text
            answer_text = re.sub(r'\\text\{[^{}]*}', '', answer_text)
            answer_value = float(answer_text)
            logging.info(f'Extracted answer from general boxed content: {answer_value}')
            return answer_value
        except ValueError:
            logging.debug(f'Found boxed content but could not convert to number: {boxed_match.group(1)}')

    logging.info('Could not extract numeric answer from model output')
    return None


def novelty_score(prompt: str, temperature=0.8, sample_size=8, solver_model_name: str = SOLVER_MODEL_NAME) -> float:
    """Higher novelty score means the model struggles more with the task (more novel/difficult).
    Calculated as 1 - (correct_solutions / total_attempts).
    """
    # Extract the expected answer from the solution part of the prompt
    answer_match = TAG_RX['answer'].search(prompt)
    if not answer_match:
        logging.warning("Couldn't find answer in prompt for novelty calculation")
        return 0.0

    expected_answer = answer_match.group(1) or answer_match.group(2)
    expected_answer = float(expected_answer)

    task_match = TAG_RX['task'].search(prompt)
    if not task_match:
        task = prompt
    else:
        task = task_match.group(1)

    correct_count = 0

    for _ in range(sample_size):
        try:
            response = solver_client.chat.completions.create(
                model=solver_model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are helpful assistant. Solve the question step-by-step. At the end, write: "The final answer is <number>".',
                    },
                    {'role': 'user', 'content': SOLVER_PROMPT.format(question=task)},
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS_RESPONSE,
            )

            completion = response.choices[0].message.content

            # Try to extract a numeric answer from the completion
            number_pattern = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
            answer_matches = re.findall(number_pattern, completion)

            if answer_matches:
                # Check all extracted numbers, with priority to those at the end
                for answer_str in reversed(answer_matches):
                    try:
                        answer = float(answer_str)
                        # Consider it correct if within small epsilon or within 1% for larger values
                        epsilon = max(1e-6, abs(expected_answer) * 0.01)
                        if abs(answer - expected_answer) < epsilon:
                            correct_count += 1
                            break
                    except ValueError:
                        continue
        except Exception as e:
            logging.warning(f'Error during novelty calculation: {e}')

    return 1.0 - (correct_count / sample_size)


def make_prediction(problem: str) -> tuple[str, dict[str, Any]]:
    """Generate a prediction for the given problem with token log probabilities.

    Returns:
        Tuple containing (generated_text, log_probabilities_info)
    """
    logging.info(f'Generating prediction for {problem[:20]}')

    prompt = BASIC_MATH_REASONING_PROMPT.format(problem=problem)

    response = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=MAX_TOKENS_RESPONSE,
        temperature=0,  # Use deterministic generation for Pass@1
        logprobs=True,  # Enable log probability tracking
    )

    generated_text = response.choices[0].text
    log_probs_data = {
        'token_logprobs': response.choices[0].logprobs.token_logprobs,
        'tokens': response.choices[0].logprobs.tokens,
    }

    logging.info(f'Generated prediction of length: {len(generated_text)}')
    return generated_text, log_probs_data


def calculate_token_log_prob_metrics(
    log_probs_data: dict[str, Any],
) -> dict[str, None] | dict[str, Any]:
    """Calculate various metrics based on token log probabilities.

    Returns:
        Dictionary with different log probability metrics
    """
    token_logprobs = log_probs_data['token_logprobs']

    filtered_logprobs = [lp for lp in token_logprobs if lp is not None]

    if not filtered_logprobs:
        return {
            'perplexity': None,
            'mean_logprob': None,
            'min_logprob': None,
            'sum_logprob': None,
            'std_logprob': None,
        }

    metrics = {
        'perplexity': math.exp(-sum(filtered_logprobs) / len(filtered_logprobs)),
        'mean_logprob': sum(filtered_logprobs) / len(filtered_logprobs),
        'min_logprob': min(filtered_logprobs),  # Most uncertain token
        'sum_logprob': sum(filtered_logprobs),  # Total log probability
        'std_logprob': np.std(filtered_logprobs),  # Standard deviation (uncertainty variability)
    }

    return metrics


def evaluate_prediction(
    predicted_answer: float | None,
    actual_answer: float,
    log_prob_metrics: dict[str, float],
) -> dict[str, Any]:
    """Evaluate the prediction against the ground truth and include log prob metrics."""
    correct = False
    if predicted_answer is not None:
        # Allow for minor numerical precision differences
        correct = abs(predicted_answer - actual_answer) < 1e-6

    result = {
        'correct': correct,
        'predicted_answer': predicted_answer,
        'actual_answer': actual_answer,
        **log_prob_metrics,
    }

    return result


def calculate_correlations(model_results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between log prob metrics and correctness."""
    metric_columns = ['perplexity', 'mean_logprob', 'min_logprob', 'sum_logprob', 'std_logprob']
    correlations = []

    model_results_df['correct_int'] = model_results_df['correct'].astype(int)

    for metric in metric_columns:
        if model_results_df[metric].isnull().any():
            continue

        pearson_corr, pearson_p = pearsonr(model_results_df[metric], model_results_df['correct_int'])
        spearman_corr, spearman_p = spearmanr(model_results_df[metric], model_results_df['correct_int'])

        correlations.append(
            {
                'metric': metric,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
            },
        )

    return pd.DataFrame(correlations)


def evaluation_pipeline():
    """Run the full evaluation pipeline."""
    gsm8k_base, easy, medium, hard = create_gsm_evaluation_datasets(to_df=True)

    # Combine all datasets with difficulty labels
    gsm8k_base['difficulty'] = 'base'
    easy['difficulty'] = 'easy'
    medium['difficulty'] = 'medium'
    hard['difficulty'] = 'hard'

    combined_df = pd.concat([gsm8k_base, easy, medium, hard])

    all_results = []

    for idx, row in combined_df.iterrows():
        question = row['question']
        solution = float(row['solution'])
        difficulty = row['difficulty']

        logging.info(f'Processing question {idx} (difficulty: {difficulty})')

        try:
            model_output, log_probs_data = make_prediction(question)

            predicted_answer = extract_answer_robust(model_output)

            log_prob_metrics = calculate_token_log_prob_metrics(log_probs_data)

            eval_result = evaluate_prediction(predicted_answer, solution, log_prob_metrics)

            result_entry = {
                'question_id': idx,
                'original_id': row['original_id'],
                'difficulty_level': difficulty,
                'question': question,
                'model_output': model_output,
                **eval_result,
            }

            all_results.append(result_entry)

        except Exception as e:
            logging.exception(f'Error processing question {idx}: {e}')

    df_result = pd.DataFrame(all_results)

    df_correlation = calculate_correlations(df_result)

    df_result.to_csv('evaluation_results.csv', index=False)
    df_correlation.to_csv('metric_correlations.csv', index=False)

    logging.info(f'Evaluation complete. Processed {len(df_result)} questions.')
    logging.info(f'Model accuracy: {df_result["correct"].mean():.2%}')
    logging.info('Metric correlations with correctness:')
    logging.info('\n' + str(df_correlation))

    return df_result, df_correlation


if __name__ == '__main__':
    results_df, correlation_df = evaluation_pipeline()
