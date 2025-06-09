import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
K_SOLVE_ATTEMPTS = 5
MAX_TOKENS_RESPONSE = 2000
MODEL_PATH = '/gpfs/home6/mmokkenstorm/sync/qwen_models/finetuned_models/n30_best/math_instruct_chattemplatetrue/lora'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FOLDER_PREFIX = '/gpfs/home6/mmokkenstorm/augmented_datasets/'
N30_FOLDER = 'n30/'
SYSTEM_PROMPT = (
    'You are a helpful assistant. Solve the math problem step by step.'
    'The answer should be a number, and you should always return it in the format: '
    'The final answer is: <answer> ANSWER </answer>.'
)
SOLVER_PROMPT = """Question: {question}\nLet's think step by step"""

# --- Regex for answer extraction ---
TAG_RX = {
    'code': re.compile(r'<code>\s*```python\s*(.*?)\s*```.*?</code>', re.DOTALL | re.IGNORECASE),
    'task': re.compile(r'<task>(.*?)</task>', re.DOTALL | re.IGNORECASE),
    'solution': re.compile(r'<solution>(.*?)</solution>', re.DOTALL | re.IGNORECASE),
    'answer': re.compile(
        r'(?:<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>'  # <answer> 24.75 </answer>
        r'|####\s*([-+]?\d+(?:\.\d+)?)\b'  # #### 24.75
        r'|The final answer is[:\s]*([-+]?\d+(?:\.\d+)?)'  # The final answer is: 24.75
        r'|\\boxed\{\s*([-+]?\d+(?:\.\d+)?)(?:\\%|%)?\s*\}'  # \boxed{24.75} or \boxed{24.75\%}
        r'|\\boxed\{\s*\\frac\{(\d+)\}\{(\d+)\}\s*\})'  # \boxed{\frac{11}{20}}
        r'|The answer is[:\s]*([-+]?\d+(?:\.\d+)?)',  # The answer is 25
        re.IGNORECASE,
    ),
}

MODEL_PATH = '/gpfs/home6/mmokkenstorm/sync/qwen_models/finetuned_models/n30_best/math_instruct_chattemplatetrue/lora'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
# --- Load model ---

model, _ = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=1024,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map='auto',
)

model = model.eval().to(DEVICE)


# --- Inference function ---
def make_prediction_transformers(question_string: str, temperature=0.8, max_new_tokens=2000):
    full_prompt = SYSTEM_PROMPT + '\n' + SOLVER_PROMPT.format(question=question_string)
    inputs = tokenizer(full_prompt, return_tensors='pt').to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )

    # Decode full output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt part from decoded output
    # This assumes the model's output begins by repeating the prompt
    if decoded.startswith(full_prompt):
        return decoded[len(full_prompt) :].strip()
    # Fallback in case the prompt isn't included exactly
    return decoded.strip()


# make_prediction_transformers('Hello, what is 2 + 2?')  # Test the model with a simple question


# --- Answer extraction ---
def extract_numeric_answer(answer: str) -> float:
    match = TAG_RX['answer'].search(answer)
    if not match:
        logging.warning(f"Couldn't find answer in text:\n{answer}")
        return 0.0
    for group in match.groups():
        if group:
            try:
                if '/' in group:
                    num, den = map(float, group.split('/'))
                    return num / den
                return float(group)
            except Exception:
                continue
    return 0.0


# --- Evaluation for one question ---
def evaluate_question(question: str, answer: str, k: int):
    expected_answer = extract_numeric_answer(answer)
    predictions, correct_predictions, extracted_answers = [], [], []

    for _ in range(k):
        try:
            completion = make_prediction_transformers(question)
            predictions.append(completion)

            extracted_answer = extract_numeric_answer(completion)
            extracted_answers.append(extracted_answer)

            epsilon = max(1e-6, abs(expected_answer) * 0.01)
            correct_predictions.append(abs(extracted_answer - expected_answer) < epsilon)
        except Exception as e:
            logging.warning(f'Error during evaluation: {e}')
            predictions.append(f'Error: {e!s}')
            correct_predictions.append(False)

    passed = any(correct_predictions)
    return {
        'question': question,
        'answer': answer,
        'passed': passed,
        'attempts': k,
        'model_path': MODEL_PATH,
        'predictions': predictions,
        'correct_predictions': correct_predictions,
        'expected_answer': expected_answer,
        'extracted_answers': extracted_answers,
        'pass_at_k': 1.0 if passed else 0.0,
        'correct_count': sum(correct_predictions),
    }


# --- Parallel evaluation ---
def run_full_evaluation_parallel(
    df: pd.DataFrame,
    k: int = 5,
    levels: list[int] = None,
    max_samples: int = None,
    batch_size: int = 1000,
    max_workers: int = 20,
) -> pd.DataFrame:
    results = []

    if levels is not None:
        df = df[df['level'].isin(levels)]

    if max_samples is not None:
        df_unique = df.groupby('source_idx').head(1)
        df = df_unique.sample(n=min(max_samples, len(df_unique)), random_state=42)

    def process_row(row):
        try:
            result = evaluate_question(row['task'], row['solution'], k=k)
            result.update(
                {
                    'source_idx': row.get('source_idx', ''),
                    'level': row.get('level', ''),
                    'n_augmented': df[df['source_idx'] == row.get('source_idx', '')].shape[0],
                    'code': row.get('code', ''),
                    'novelty': row.get('novelty', ''),
                    'difficulty': row.get('difficulty', ''),
                },
            )
            return result
        except Exception as e:
            logging.warning(f'Failed row {row.get("source_idx")}: {e}')
            return None

    for i in range((len(df) + batch_size - 1) // batch_size):
        batch = df.iloc[i * batch_size : (i + 1) * batch_size]
        logging.info(f'Processing batch {i + 1} of {len(df) // batch_size + 1}')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_row, row) for _, row in batch.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)

    return pd.DataFrame(results)


#
# # --- Run evaluation ---
# if __name__ == '__main__':
#     df = pd.read_csv(FOLDER_PREFIX + N30_FOLDER + 'augmented_best.csv')
#     df = df.rename(columns={df.columns[0]: 'Unnamed: 0'})  # optional cleanup
#     df['source_idx'] = df.get('source_idx', df.index)
#     df['level'] = df.get('level', 0)
#
#     df = df[['source_idx', 'level', 'code', 'task', 'solution', 'novelty', 'difficulty']]
#     eval_df = run_full_evaluation_parallel(
#         df=df[:1],
#         k=K_SOLVE_ATTEMPTS,
#         levels=None,
#         max_samples=None,
#         max_workers=20,
#     )
#
#     eval_df['pass_percentage'] = eval_df['correct_count'] / eval_df['attempts'] * 100
#     eval_df.to_csv(FOLDER_PREFIX + N30_FOLDER + 'augmented_best_eval_result_finetuned.csv', index=False)

if __name__ == '__main__':
    # Load additional datasets
    from actual_project.pipelines.create_gsm_evaluation_datasets import (
        create_gsm_evaluation_datasets,  # replace with actual import if needed
    )

    selected_gsm8k, selected_easy, selected_medium, selected_hard = create_gsm_evaluation_datasets(to_df=True)

    datasets_to_process = [
        # (pd.read_csv(FOLDER_PREFIX + 'all_df_256.csv', header=None), 'df_n256_all'),
        (pd.read_csv(FOLDER_PREFIX + 'best_df_256.csv'), 'df_n256_best'),
        (selected_gsm8k, 'selected_gsm8k'),
        (selected_easy, 'selected_easy'),
        (selected_medium, 'selected_medium'),
        (selected_hard, 'selected_hard'),
    ]

    all_results = []

    for df, name in datasets_to_process:
        logging.info(f'Evaluating dataset: {name}')
        df = df.rename(columns={df.columns[0]: 'Unnamed: 0'}) if 'Unnamed: 0' not in df.columns else df
        df['source_idx'] = df.get('source_idx', df.index)
        df['level'] = df.get('level', 0)

        df = df[['source_idx', 'level', 'code', 'task', 'solution', 'novelty', 'difficulty']]
        levels = None
        eval_df = run_full_evaluation_parallel(
            df=df,
            k=K_SOLVE_ATTEMPTS,
            levels=levels,
            max_samples=None,
            max_workers=1,
        )

        eval_df['pass_percentage'] = eval_df['correct_count'] / eval_df['attempts'] * 100
        eval_df['dataset'] = name
        all_results.append(eval_df)

        eval_df.to_csv(FOLDER_PREFIX + f'{name}_eval_result_finetuned.csv', index=False)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(FOLDER_PREFIX + 'final_eval_results_finetuned.csv', index=False)
    logging.info('Saved final evaluation results.')
