# import logging
# import re
#
# import torch
# from transformers import AutoTokenizer
# from unsloth import FastLanguageModel
#
# # --- Constants ---
#
#
# logging.basicConfig(
#     level=logging.DEBUG,  # or DEBUG for even more verbosity
#     format='[%(levelname)s] %(message)s',
# )
# logger = logging.getLogger(__name__)
# K_SOLVE_ATTEMPTS = 5
# MAX_TOKENS_RESPONSE = 2000
# MODEL_PATH = '/home/mmokkenstorm/sync/qwen_models/finetuned_models/n30_best/math_instruct_chattemplatetrue_v2/lora'
# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# logger.info(f'Loading model from {MODEL_PATH}')
# FOLDER_PREFIX = '/gpfs/home6/mmokkenstorm/augmented_datasets/'
# N30_FOLDER = 'n30/'
# SYSTEM_PROMPT = (
#     'You are a helpful assistant. Solve the math problem step by step.'
#     'The answer should be a number, and you should always return it in the format: '
#     'The final answer is: <answer>  </answer>.'
# )
# FEWSHOT_EXAMPLES = """
# Example questions
# Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?
# Let's think step by step
# In 2005, 60/2=30 kids came to the cookout.
# In 2006, 30*2/3=20 kids came to the cookout.
# The answer is 20
#
# """.strip()
#
# QUESTION_TEMPLATE = """
#
# Question to answer: {question}
# Let's think step by step
# """.strip()
#
# # Then inside your batch_generate_transformers function:
#
# # --- Regex for answer extraction ---
# TAG_RX = {
#     'code': re.compile(r'<code>\s*```python\s*(.*?)\s*```.*?</code>', re.DOTALL | re.IGNORECASE),
#     'task': re.compile(r'<task>(.*?)</task>', re.DOTALL | re.IGNORECASE),
#     'solution': re.compile(r'<solution>(.*?)</solution>', re.DOTALL | re.IGNORECASE),
#     'answer': re.compile(
#         r'(?:<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>'
#         r'|####\s*([-+]?\d+(?:\.\d+)?)\b'
#         r'|The final answer is[:\s]*([-+]?\d+(?:\.\d+)?)'
#         r'|\\boxed\{\s*([-+]?\d+(?:\.\d+)?)(?:\\%|%)?\s*\}'
#         r'|\\boxed\{\s*\\frac\{(\d+)\}\{(\d+)\}\s*\})'
#         r'|The answer is[:\s]*([-+]?\d+(?:\.\d+)?)',
#         re.IGNORECASE,
#     ),
# }
#
# # --- Load tokenizer and model on single GPU ---
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
#
# model, _ = FastLanguageModel.from_pretrained(
#     MODEL_PATH,
#     max_seq_length=1024,
#     dtype=torch.bfloat16,
#     load_in_4bit=True,
#     device_map={'': DEVICE},  # Put entire model on cuda:0
# )
#
# model.eval()
#
#
# # --- Inference ---
# def batch_generate_transformers(question_list, temperature=0.8, max_new_tokens=2000):
#     # prompts = [SYSTEM_PROMPT + '\n' + SOLVER_PROMPT.format(question=q) for q in question_list]
#
#     QUESTION_TEMPLATE.format(question=question_list[0])
#     prompts = [f'{SYSTEM_PROMPT}\n{FEWSHOT_EXAMPLES}\n{QUESTION_TEMPLATE.format(question=q)}' for q in question_list]
#     print(prompts)
#     inputs = tokenizer(
#         prompts,
#         return_tensors='pt',
#         padding=True,
#         truncation=True,
#         max_length=1024,
#     ).to(model.device)
#
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=2000,
#             temperature=0.8,
#             do_sample=True,
#         )
#
#     decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     # Strip full prompt from each output
#     stripped_outputs = [
#         decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
#         for decoded, prompt in zip(decoded_outputs, prompts, strict=False)
#     ]
#     return stripped_outputs
#
#
# # SAMPLE_QUESTION = 'A pet store sells two types of fish tanks. One type has a capacity of 20 liters, with 7 liters filled with water, and another type has a capacity of 30 liters, with 9 liters filled with water. What is the combined percentage of both tank capacities that are currently filled?'
# # question_list = [SAMPLE_QUESTION]
# # ✅ Test
#
# # print(batch_generate_transformers([SAMPLE_QUESTION]))
#
#
# # --- Answer extraction ---
# def extract_numeric_answer(answer: str) -> float:
#     match = TAG_RX['answer'].search(answer)
#     if not match:
#         logging.warning(f"Couldn't find answer in text:\n{answer}")
#         return 0.0
#     for group in match.groups():
#         if group:
#             try:
#                 if '/' in group:
#                     num, den = map(float, group.split('/'))
#                     return num / den
#                 return float(group)
#             except Exception:
#                 continue
#     return 0.0
#
#
# def is_correct(pred: float, answer: str | float, tol: float = 1e-2) -> bool:
#     """Checks if predicted numeric answer is within `tol` of the ground truth.
#     Accepts answer as string or float.
#     """
#     try:
#         true_answer = float(answer)
#         return abs(pred - true_answer) < tol
#     except (ValueError, TypeError):
#         return False
#
#
# # --- Evaluation for one question ---
# def evaluate_batch(questions, answers, k=5):
#     import numpy as np  # You may have forgotten this import
#
#     batched_questions = []
#     index_map = []
#
#     # Prepare k prompts per question
#     for i, question in enumerate(questions):
#         for _ in range(k):
#             batched_questions.append(question)
#             index_map.append(i)
#
#     # Generate outputs
#     all_outputs = batch_generate_transformers(batched_questions)
#
#     # Group outputs back to questions
#     completions_per_question = [[] for _ in range(len(questions))]
#     for output, i in zip(all_outputs, index_map, strict=False):
#         completions_per_question[i].append(output)
#
#     # Process answers
#     result_dicts = []
#     for i, (q, a) in enumerate(zip(questions, answers, strict=False)):
#         answer = a if isinstance(a, str) else a['answer']
#         preds = completions_per_question[i]
#         extracted = [extract_numeric_answer(p) for p in preds]
#         is_corrects = [is_correct(e, answer) for e in extracted]
#
#         result_dicts.append(
#             {
#                 'original_id': q['original_id'] if isinstance(q, dict) and 'original_id' in q else None,
#                 'question': q,
#                 'answer': answer,
#                 'predictions': preds,
#                 'is_corrects': is_corrects,
#                 'correct_frac': np.mean(is_corrects),
#                 'unique_answers': len(set(extracted)),
#                 'answers': extracted,
#             },
#         )
#
#     return result_dicts
#
#
# import pandas as pd
#
#
# # --- Parallel evaluation ---
# def run_full_evaluation_batched(
#     df: pd.DataFrame,
#     k: int = 5,
#     levels: list[int] = None,
#     max_samples: int = None,
#     batch_size: int = 16,
# ) -> pd.DataFrame:
#     results = []
#
#     if levels is not None:
#         df = df[df['level'].isin(levels)]
#
#     if max_samples is not None:
#         df_unique = df.groupby('source_idx').head(1)
#         df = df_unique.sample(n=min(max_samples, len(df_unique)), random_state=42)
#
#     for i in range(0, len(df), batch_size):
#         batch = df.iloc[i : i + batch_size]
#         logging.warning(f'Processing batch {i // batch_size + 1} of {len(df) // batch_size + 1}')
#
#         questions = batch['task'].tolist()
#         answers = batch['solution'].tolist()
#
#         batch_results = evaluate_batch(questions, answers, k=batch_size)
#
#         # Attach metadata
#         for res, (_, row) in zip(batch_results, batch.iterrows(), strict=False):
#             res.update(
#                 {
#                     'source_idx': row.get('source_idx', ''),
#                     'level': row.get('level', ''),
#                     'n_augmented': df[df['source_idx'] == row.get('source_idx', '')].shape[0],
#                     'code': row.get('code', ''),
#                     'novelty': row.get('novelty', ''),
#                     'difficulty': row.get('difficulty', ''),
#                 },
#             )
#             results.append(res)
#
#     return pd.DataFrame(results)
#
#
# #
# # # --- Run evaluation ---
# # if __name__ == '__main__':
# #     df = pd.read_csv(FOLDER_PREFIX + N30_FOLDER + 'augmented_best.csv')
# #     df = df.rename(columns={df.columns[0]: 'Unnamed: 0'})  # optional cleanup
# #     df['source_idx'] = df.get('source_idx', df.index)
# #     df['level'] = df.get('level', 0)
# #
# #     df = df[['source_idx', 'level', 'code', 'task', 'solution', 'novelty', 'difficulty']]
# #     eval_df = run_full_evaluation_parallel(
# #         df=df[:1],
# #         k=K_SOLVE_ATTEMPTS,
# #         levels=None,
# #         max_samples=None,
# #         max_workers=20,
# #     )
# #
# #     eval_df['pass_percentage'] = eval_df['correct_count'] / eval_df['attempts'] * 100
# #     eval_df.to_csv(FOLDER_PREFIX + N30_FOLDER + 'augmented_best_eval_result_finetuned.csv', index=False)
#
# if __name__ == '__main__':
#     # Load additional datasets
#     # from actual_project.pipelines.create_gsm_evaluation_datasets import (
#     #     create_gsm_evaluation_datasets,  # replace with actual import if needed
#     # )
#
#     selected_gsm8k, selected_easy, selected_medium, selected_hard = create_gsm_evaluation_datasets(to_df=True)
#     for df in [selected_gsm8k, selected_easy, selected_medium, selected_hard]:
#         df['source_idx'] = df.get('source_idx', df.index)
#         df['level'] = df.get('level', 0)
#         df['task'] = df['question']  # Ensure 'task' column exists
#         df['solution'] = df['answer']
#     datasets_to_process = [
#         # (pd.read_csv(FOLDER_PREFIX + 'all_df_256.csv', header=None), 'df_n256_all'),
#         (pd.read_csv(FOLDER_PREFIX + 'best_df_256.csv'), 'df_n256_best'),
#         (selected_gsm8k, 'selected_gsm8k'),
#         (selected_easy, 'selected_easy'),
#         (selected_medium, 'selected_medium'),
#         (selected_hard, 'selected_hard'),
#     ]
#
#     all_results = []
#
#     for df, name in datasets_to_process:
#         logging.warning(f'Evaluating dataset: {name}')
#         df = df.rename(columns={df.columns[0]: 'Unnamed: 0'}) if 'Unnamed: 0' not in df.columns else df
#
#         # df = df[['source_idx', 'level', 'code', 'task', 'solution', 'novelty', 'difficulty']]
#         levels = None
#         eval_df = run_full_evaluation_batched(
#             df=df,
#             k=K_SOLVE_ATTEMPTS,
#             levels=levels,
#             max_samples=None,
#             batch_size=16,
#         )
#         eval_df['correct_counts'] = eval_df['is_corrects'].apply(lambda x: sum(x))
#         eval_df['pass_percentage'] = eval_df['correct_counts'] / eval_df['attempts'] * 100
#         eval_df['dataset'] = name
#         all_results.append(eval_df)
#
#         eval_df.to_csv(FOLDER_PREFIX + f'{name}_eval_result_finetuned.csv', index=False)
#
#     final_df = pd.concat(all_results, ignore_index=True)
#     final_df.to_csv(FOLDER_PREFIX + 'final_eval_results_finetuned.csv', index=False)
#     logging.warning('Saved final evaluation results.')
