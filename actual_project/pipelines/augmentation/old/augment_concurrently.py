import logging
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai
import pandas as pd
from openai import OpenAI

SOLVER_MODEL_NAME = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
BASE_URL_SOLVER = 'http://localhost:8001/v1'
AUGMENTER_MODEL_NAME = 'Qwen/Qwen2.5-Coder-7B-Instruct'
BASE_URL_AUGMENTER = 'http://localhost:8000/v1'

N_AUGS_PER_SOURCE = 10  # how many *levels* of augmentation per task
SAMPLE_PER_AUG = 10  # how many candidate generations at each level

MAX_TOKENS_RESPONSE = 2000

TEMPERATURE = 0.8
MIN_NOVELTY = 0  # threshold; higher = more novel
MIN_DIFFICULTY = 0  # threshold; higher = harder
MAX_API_RETRIES = 3

REASONS = {
    'ACCEPTED': 'accepted',
    'INCORRECT': 'incorrect_code',
    'LOW_NOVELTY': 'too_low_novelty',
    'LOW_DIFFICULTY': 'too_low_difficulty',
}

solver_client = OpenAI(base_url=BASE_URL_SOLVER, api_key='EMPTY')
augmenter_client = OpenAI(base_url=BASE_URL_AUGMENTER, api_key='EMPTY')

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

VALID_BLOCK_TEMPLATE = (
    '<code>```python\n$PY\n```</code>\n<task>$TASK</task>\n<solution>$SOL #### <answer> $ANS </answer></solution>'
)

SOLVER_PROMPT = """Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?
Let's think step by step
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30*2/3=20 kids came to the cookout.
The answer is 20

Question: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?
Let's think step by step
Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.
The total monthly earning of Zilla is represented by 100%, so $19 × 100 = $1900 is her monthly earnings.
So, $1900/2 = $950 is spent on her other monthly expenses.
The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.
Hence, she saves $1900 - $1083 = $817 per month.
The answer is 817

Question: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz’s ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?
Let's think step by step
The total ratio representing the slices of pizza that Buzz bought is 5+8=13
If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices
Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28
The answer is 28

Question: Jame gets a raise to $20 per hour and works 40 hours a week. His old job was $16 an hour for 25 hours per week. How much more money does he make per year in his new job than the old job if he works 52 weeks a year?
Let's think step by step
He makes 20*40=$800 per week
He used to make 16*25=$400 per week
So his raise was 800-400=$400 per week
So he makes 400*52=$20,800 per year more
The answer is 20800

Question: Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive?
Let's think step by step
Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats
Each student will receive 80 / 20 = 4 sweet treats
The answer is 4

Question: A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot’s vehicles in all?
Let's think step by step
The used car lot has 24 / 3 = 8 motorcycles with 2 tires each.
The lot has 24 - 8 = 16 cars for sale
There are 16 / 4 = 4 cars with a spare tire with 5 tires each.
The lot has 16 - 4 = 12 cars with 4 tires each.
Thus, the used car lot’s vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all.
The answer is 84

Question: Norma takes her clothes to the laundry. She leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds 3 sweaters and triple the number of T-shirts. How many items are missing?
Let's think step by step
Norma left 9 T-shirts
And twice as many sweaters, she took 9 * 2= 18 sweaters
Adding the T-shirts and sweaters, Norma left 9 + 18 = 27 clothes
When she came back, she found 3 sweaters
And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts
Adding the T-shirts and sweaters, Norma found 3 + 9 = 12 clothes
Subtracting the clothes she left from the clothes she found, 27 - 12 = 15 clothes are missing
The answer is 15

Question: Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard?
Let's think step by step
During 30 days Adam picked 4 * 30 = 120 apples.
So in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard.
The answer is 350

Question: {question}
Let's think step by step"""


difficulty = 0
SYSTEM_PROMPT_TEMPLATE = rf"""
You are an **Augmenter** that turns a quantitative word-problem into a
*strictly harder* variant.

──────────────── INPUT (always mutually consistent) ────────────────
{VALID_BLOCK_TEMPLATE.replace('$PY', '…').replace('$TASK', '…').replace('$SOL', '…').replace('$ANS', '…')}

──────────────── OUTPUT (exactly ONE harder variant) ───────────────
{
    VALID_BLOCK_TEMPLATE.replace('$PY', '$NEW_PYTHON_CODE')
    .replace('$TASK', '$NEW_PROBLEM')
    .replace('$SOL', '$NEW_SOLUTION')
    .replace('$ANS', '$NUMERIC_ANSWER')
}

━━━━━━━━━━━  RULES (read carefully)  ━━━━━━━━━━━
1. **Return only the block above** – no extra text.
2. Copy every delimiter literally:  
   `<code>`, triple-back-tick python fence, `</code>`,  
   `<task> … </task>`, `<solution> … </solution>`,  
   four # characters, `<answer> … </answer>`.
3. *Harder* means a larger reasoning load – not just bigger numbers.
   Let **K = {difficulty}**.  Your new problem **must satisfy**:
      • At least **K + 1 arithmetic/logic steps** in the solution.  
      • Introduce **min(K, 4) of these complexity devices**  
        (distinct from the input):  
          – Fractions / decimals / percentages / ratios  
          – Unit conversions or rates (speed, price per unit, etc.)  
          – Conditional statements (“if … then …”) or comparisons  
          – Multi-entity relations (ages, mixtures, work-rates, etc.)  
          – Distractor quantities that are *not* needed for the answer  
          – Re-using an intermediate result in a later step  
      • The final numeric answer **must differ** from the source.  
4. Do **not** copy nouns, story setting, or wording from the input.  
   Change characters, context, and phrasing.  
5. Preserve internal consistency: the `<code>` output must `print`
   exactly the number inside `<answer>` when executed.  
6. Avoid unnecessarily large or tiny numbers; keep them readable.  
7. If you reference units (kg, km/h, dollars, minutes, etc.), keep
   them consistent throughout code, task, and solution.  
8. Never mention these instructions or the value of **K**.\

You can add your reasoning steps In the <solution> </solution> block.
"""


TEXT2CODE_SYSTEM_PROMPT = """Convert the <task> and <solution> below into working Python
that converts the question-answer pair into Python code.
Wrap the code in <code>...</code> tags and triple-back-ticked Python so downstream parsers can extract it.
Don't add any extra text or comments.
The code should be valid and executable.
"""

RX_CODE_BLOCK = re.compile(r'<code>.*?```python(.*?)```.*?</code>', re.DOTALL)
RX_TASK_BLOCK = re.compile(r'<task>(.*?)</task>', re.DOTALL)
RX_SOL_BLOCK = re.compile(r'<solution>(.*?)</solution>', re.DOTALL)
RX_FINAL_ANSWER = re.compile(r'####\s*([+-]?\d+(?:\.\d+)?)')

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


def _strip(text: str) -> str:
    return text.strip(' \n')


def safe_chat_completion(
    messages: list[dict],
    backoff_base: int = 1,
    presence_penalty: float = 0.6,
    frequency_penalty: float = 0.4,
    **kwargs,
) -> Any:
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = chat_completion(
                messages,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                **kwargs,
            )
            logging.debug('Response: %s', response)
            return response

        except openai.OpenAIError as e:
            if attempt == MAX_API_RETRIES:
                raise
            wait = backoff_base * 2 ** (attempt - 1) * random.uniform(0.8, 1.2)
            logging.warning(
                'API error (%s). Retry %d/%d after %.1fs',
                e.__class__.__name__,
                attempt,
                MAX_API_RETRIES,
                wait,
            )
            time.sleep(wait)


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


def extract_blocks(text: str) -> tuple[str, str, str]:
    m_code = TAG_RX['code'].search(text)
    if not m_code:
        fences = re.findall(r'```python\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if fences:
            code = fences[-1].strip()
        else:
            raise ValueError('no Python code block found')
    else:
        code = m_code.group(1).strip()

    m_task = TAG_RX['task'].search(text)
    m_sol = TAG_RX['solution'].search(text)
    if not (m_task and m_sol):
        raise ValueError('missing <task> or <solution>')

    return code, m_task.group(1).strip(), m_sol.group(1).strip()


def extract_answer(solution: str) -> float:
    m = TAG_RX['answer'].search(solution)
    if not m:
        raise ValueError('no numeric answer found')
    num_str = m.group(1) or m.group(2)
    return float(num_str)


def dedup_answer_tags(text: str) -> str:
    def _dedup(m):
        seen = False

        def inner(ans_match):
            nonlocal seen
            if seen:
                return ''
            seen = True
            return ans_match.group(0)

        return re.sub(r'<answer>.*?</answer>', inner, m.group(0), flags=re.DOTALL)

    return re.sub(r'<solution>.*?</solution>', _dedup, text, flags=re.DOTALL)


def extract_code_task_solution(text: str) -> tuple[str, str, str]:
    code_match = re.search(r'<code>.*?```python(.*?)```(?:\s*```)?\s*</code>', text, re.DOTALL)
    if code_match:
        code = re.sub(r'^```python\s*', '', code_match.group(1)).strip()
    else:
        fenced_blocks = re.findall(r'```python\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        code = fenced_blocks[-1].strip() if fenced_blocks else ''

    task_match = RX_TASK_BLOCK.search(text)
    sol_match = RX_SOL_BLOCK.search(text)

    task = _strip(task_match.group(1)) if task_match else ''
    sol = _strip(sol_match.group(1)) if sol_match else ''

    return code, task, sol


def run_code(code: str) -> str:
    with tempfile.NamedTemporaryFile('w+', suffix='.py') as tmp:
        tmp.write(code)
        tmp.flush()
        proc = subprocess.run(['python', '-E', '-I', tmp.name], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[:300])
        return proc.stdout.strip()


def code_matches_solution(code: str, solution: str) -> bool:
    try:
        executed_code_result = run_code(code)

    except Exception as e:
        logging.exception('Code execution error: %s', e)
        return False

    if isinstance(executed_code_result, str):
        extracted_code_solution = re.search(r'\b\d+(?:\.\d+)?\b', executed_code_result)
        if not extracted_code_solution:
            return False
        extracted_code_solution = float(extracted_code_solution.group())
    else:
        extracted_code_solution = executed_code_result

    extracted_actual_answer = extract_answer(solution)
    extracted_code_solution = round(extracted_code_solution, 3)
    return extracted_code_solution == round(extracted_actual_answer, 3)


def chat_completion(
    messages: list[dict],
    *,
    log_probs: bool = False,
    number_of_generated_outputs: int = 1,
    **kwargs,
) -> Any:
    if log_probs:
        kwargs['logprobs'] = True
    kwargs['n'] = max(1, number_of_generated_outputs)
    response = augmenter_client.chat.completions.create(
        model=AUGMENTER_MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_RESPONSE,
        **kwargs,
    )
    if number_of_generated_outputs == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]


def text2code(task: str, solution: str) -> str | None:
    messages = [
        {'role': 'system', 'content': TEXT2CODE_SYSTEM_PROMPT},
        {'role': 'user', 'content': f'<task>{task}</task>\n<solution>{solution}</solution>'},
    ]
    resp = safe_chat_completion(messages)

    try:
        resp = enforce_format(resp)
        code, *_ = extract_blocks(resp)
    except Exception as exc:
        logging.exception('text2code formatting/extraction failed: %s', exc)
        return None

    if not code_matches_solution(code, solution):
        logging.info('text2code produced non‑matching code.')
        return None
    return code


def evaluate_candidate(code: str, task: str, sol: str) -> dict[str, Any]:
    """Evaluate a single candidate augmentation and return a dict with metrics."""
    result: dict[str, Any] = {
        'code': code,
        'task': task,
        'solution': sol,
        'novelty': None,
        'difficulty': None,
        'is_correct': False,
        'reason': REASONS['INCORRECT'],
    }

    correct = code_matches_solution(code, sol)
    result['is_correct'] = correct

    if not correct:
        result['reason'] = REASONS['INCORRECT']
        return result

    # Only compute novelty & difficulty if correct (saves API calls)
    novelty = novelty_score(f'{task}\n{sol}')
    # difficulty_val = answer_difficulty(task, sol)
    result['novelty'] = novelty
    # result['difficulty'] = difficulty_val

    if abs(novelty) < MIN_NOVELTY:
        result['reason'] = REASONS['LOW_NOVELTY']
    # elif abs(difficulty_val) < MIN_DIFFICULTY:
    #     result['reason'] = REASONS['LOW_DIFFICULTY']
    else:
        result['reason'] = REASONS['ACCEPTED']

    return result


def augment_once(
    base_code: str,
    base_task: str,
    base_sol: str,
    task_difficulty_level: int,
    max_samples: int = SAMPLE_PER_AUG,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Generate up to `max_samples` harder variants and pick the best accepted one.

    Returns (best_candidate_or_None, all_attempted_candidates)
    """
    try:
        numeric_answer = extract_answer(base_sol).__str__()
    except Exception as e:
        logging.exception('Failed to extract numeric answer: %s', e)
        return None, []
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(difficulty=task_difficulty_level)
    user_block = (
        VALID_BLOCK_TEMPLATE.replace('$PY', base_code)
        .replace('$TASK', base_task)
        .replace('$SOL', base_sol)
        .replace('$ANS', numeric_answer)
    )

    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_block},
    ]
    logging.info('Generating %d samples', max_samples)
    replies: list[str] = safe_chat_completion(messages, number_of_generated_outputs=max_samples)
    attempted: list[dict[str, Any]] = []

    best_candidate: dict[str, Any] | None = None
    accepted_candidates: list[dict[str, Any]] = []

    for raw_reply in replies:
        try:
            raw_reply = dedup_answer_tags(raw_reply)
            raw_reply = enforce_format(raw_reply)
            new_code, new_task, new_sol = extract_blocks(raw_reply)
            logging.debug('new_code: %s', new_code)
        except Exception as e:
            logging.exception('Format/extraction error: %s', e)
            continue

        try:
            cand = evaluate_candidate(new_code, new_task, new_sol)
        except Exception as e:
            logging.exception('Candidate evaluation failed: %s', e)
            continue

        attempted.append(cand)

        if cand['reason'] == REASONS['ACCEPTED']:
            accepted_candidates.append(cand)

        if accepted_candidates:
            logging.debug('Accepted candidates: %s', accepted_candidates)
            best_candidate = max(
                accepted_candidates,
                key=lambda c: (
                    c['novelty']
                ),  # sort by novelty, take the Max value --> this indicated that the model struggled the most
            )
            logging.debug('Best candidate: %s', best_candidate)
        else:
            best_candidate = None

            logging.debug('No accepted candidates: %s', accepted_candidates)
    return best_candidate, attempted


def enforce_format(raw: str, retries: int = 2) -> str:
    from textwrap import dedent

    for _ in range(retries + 1):
        try:
            extract_blocks(raw)
            return raw
        except ValueError:
            fix_prompt = [
                {
                    'role': 'system',
                    'content': dedent(f"""
                    Your job is to repair ONLY the formatting of the text the
                    user gives you so that it exactly matches this schema:

                    {VALID_BLOCK_TEMPLATE}

                    Do not change any numbers, wording, or variable names.
                    Return *only* the corrected block.
                """),
                },
                {'role': 'user', 'content': raw},
            ]
            raw = safe_chat_completion(fix_prompt)
    raise ValueError('Unable to repair formatting after retries')


def augment_one_row(idx: int, task: str, solution: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Performs all augmentations for a single row."""
    try:
        base_code = text2code(task, solution)
    except Exception as e:
        logging.exception('[%s] text2code failed: %s', idx, e)
        return [], []

    if base_code is None:
        logging.info('[%s] Failed to obtain base code; skipping.', idx)
        return [], []

    best_rows = []
    all_rows = []

    curr_code, curr_task, curr_sol = base_code, task, solution

    for level in range(1, N_AUGS_PER_SOURCE + 1):
        logging.info('[%s] Augmenting level %d', idx, level + 1)
        best_candidate, attempts = augment_once(curr_code, curr_task, curr_sol, task_difficulty_level=level + 1)

        for cand in attempts:
            all_rows.append({'source_idx': idx, 'level': level, **cand})

        if best_candidate is not None:
            best_rows.append({'source_idx': idx, 'level': level, **best_candidate})
            curr_code, curr_task, curr_sol = (
                best_candidate['code'],
                best_candidate['task'],
                best_candidate['solution'],
            )
        else:
            logging.info('[%s] No accepted candidate at level %d; stopping.', idx, level + 1)
            break

    return best_rows, all_rows


def augment_dataframe(
    df: pd.DataFrame,
    max_concurrent: int = 10,
    checkpoint_every: int = 100,
    checkpoint_dir: str = '/home/mmokkenstorm/sync/outputs/augmentation_output/evaluation_checkpoints',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Augment the input DataFrame concurrently and return (best_df, all_attempts_df).
    Saves checkpoint every `checkpoint_every` examples.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_rows = []
    all_rows = []
    with ThreadPoolExecutor(max_concurrent) as executor:
        futures = {
            executor.submit(augment_one_row, idx, row['question'], row['answer']): idx for idx, row in df.iterrows()
        }

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                best, all_ = future.result()
                best_rows.extend(best)
                all_rows.extend(all_)
                completed += 1

                if completed % checkpoint_every == 0:
                    best_df = pd.DataFrame(best_rows)
                    all_df = pd.DataFrame(all_rows)

                    best_ckpt = os.path.join(checkpoint_dir, f'checkpoint_best_{completed}.csv')
                    all_ckpt = os.path.join(checkpoint_dir, f'checkpoint_all_{completed}.csv')

                    best_df.to_csv(best_ckpt, index=False)
                    all_df.to_csv(all_ckpt, index=False)

                    logging.info(f'Checkpoint saved at {completed} examples: {best_ckpt}, {all_ckpt}')

            except Exception as e:
                logging.exception('Row augmentation failed (idx=%s): %s', idx, e)

    return pd.DataFrame(best_rows), pd.DataFrame(all_rows)


if __name__ == '__main__':
    import os
    import sys

    sys.path.extend(
        [
            '/gpfs/home6/mmokkenstorm/tmp/complex_task_gen/',
            '/tmp/ChIfXZallM',
            '/home/mmokkenstorm/tmp/complex_task_gen/actual_project',
        ],
    )
    from actual_project.pipelines.gsm_evaluation_dataset_creation import create_gsm_evaluation_datasets

    gsm8k, gsm_easy, gsm_med, gsm_hard = create_gsm_evaluation_datasets(to_df=True)
    logging.info(os.getcwd())
    # gsm8k_dataset = load_dataset('openai/gsm8k', 'main', split='train')
    gsm8k_train = pd.DataFrame(gsm8k)

    best_df, all_df = augment_dataframe(df=gsm8k_train[40:], max_concurrent=10, checkpoint_every=10)

    best_df.to_csv('augmented_best_subset_new_gsm8k_eval_10.csv', index=False)
    all_df.to_csv('augmented_all_subset_new_gsm8k_eval_10.csv', index=False)

    logging.info('Saved augmented_best.csv and augmented_all.csv')
