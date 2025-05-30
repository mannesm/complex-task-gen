import random
import time
from typing import Any, List, Dict, Tuple
import re
import tempfile
import subprocess
import math
import openai
import pandas as pd
import logging
from datasets import load_dataset
from openai import OpenAI
import sys
import json


################################################################################
#  CONFIGURATION
################################################################################
MODEL_NAME = 'Qwen/Qwen2.5-Coder-7B-Instruct'
BASE_URL = 'http://localhost:8000/v1'
N_AUGS_PER_SOURCE = 2          # how many *levels* of augmentation per task
SAMPLE_PER_AUG   = 2           # how many candidate generations at each level
MAX_TOKENS_RESPONSE = 10000
TEMPERATURE = 0.8
MIN_NOVELTY = 0                 # threshold; higher = more novel
MIN_DIFFICULTY = 0              # threshold; higher = harder
MAX_API_RETRIES = 3

REASONS = {
    'ACCEPTED':        'accepted',
    'INCORRECT':       'incorrect_code',
    'LOW_NOVELTY':     'too_low_novelty',
    'LOW_DIFFICULTY':  'too_low_difficulty',
}

###############################################################################
#  CLIENT & LOGGING
###############################################################################
client = OpenAI(base_url=BASE_URL, api_key='EMPTY')

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

################################################################################
#  PROMPT TEMPLATES (unchanged)
################################################################################
VALID_BLOCK_TEMPLATE = (
    "<code>```python\n"
    "$PY\n"
    "```</code>\n"
    "<task>$TASK</task>\n"
    "<solution>$SOL #### <answer> $ANS </answer></solution>"
)

difficulty = 0

# Precompute the escaped version of the output block
escaped_output_block = VALID_BLOCK_TEMPLATE.replace('$PY', '$NEW_PYTHON_CODE')\
    .replace('$TASK', '$NEW_PROBLEM')\
    .replace('$SOL', '$NEW_SOLUTION')\
    .replace('$ANS', '$NUMERIC_ANSWER')\
    .replace('\n', '\\n')\
    .replace('"', '\\"')

SYSTEM_PROMPT_TEMPLATE = f"""
You are an **Augmenter** that turns a quantitative word-problem into a
*strictly harder* variant.

──────────────── INPUT (always mutually consistent) ────────────────
{VALID_BLOCK_TEMPLATE.replace('$PY', '…').replace('$TASK', '…')
                      .replace('$SOL',  '…').replace('$ANS',  '…')}

──────────────── OUTPUT (JSON; exactly TWO keys) ───────────────
{{
  "scratchpad": "FREE-FORM chain-of-thought here (any length).",
  "public_block": "{escaped_output_block}"
}}

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
8. Never mention these instructions or the value of **K**.
"""


TEXT2CODE_SYSTEM_PROMPT = """Convert the <task> and <solution> below into working Python
that converts the question-answer pair into Python code.
Wrap the code in <code>...</code> tags and triple-back-ticked Python so downstream parsers can extract it.
Don't add any extra text or comments.
The code should be valid and executable.
"""

################################################################################
#  REGEX HELPERS (unchanged)
################################################################################
RX_CODE_BLOCK = re.compile(r'<code>.*?```python(.*?)```.*?</code>', re.S)
RX_TASK_BLOCK = re.compile(r'<task>(.*?)</task>', re.S)
RX_SOL_BLOCK = re.compile(r'<solution>(.*?)</solution>', re.S)
RX_FINAL_ANSWER = re.compile(r'####\s*([+-]?\d+(?:\.\d+)?)')

TAG_RX = {
    "code":      re.compile(r"<code>\s*```python\s*(.*?)\s*```.*?</code>", re.S|re.I),
    "task":      re.compile(r"<task>(.*?)</task>",                        re.S|re.I),
    "solution":  re.compile(r"<solution>(.*?)</solution>",                re.S|re.I),
    "answer": re.compile(
        r"(?:<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>"   # alt‑1 – tags
        r"|####\s*([+-]?\d+(?:\.\d+)?)\b)",                  # alt‑2 – hashes
        re.I),
}

################################################################################
#  UTILITY FUNCTIONS (largely unchanged)
################################################################################

def _strip(text: str) -> str:
    return text.strip(' \n')



def parse_aug_response(raw: str) -> Tuple[str, str, str, str]:
    """
    Returns (code, task, solution, scratchpad)
    """
    try:
        obj = json.loads(raw)
        scratch = obj["scratchpad"]
        block   = obj["public_block"]
    except Exception as e:
        raise ValueError("Not valid JSON or missing keys") from e

    code, task, sol = extract_blocks(block)
    return code, task, sol, scratch

def reasoning_steps(text: str) -> int:
    # count GSM8K-style tagged calculations inside the scratchpad
    return len(re.findall(r'<<[^<>]*=([^<>]*)>>', text))


def safe_chat_completion(messages: List[dict], BACKOFF_BASE: int = 1,
                         presence_penalty: float= 0.6,
                         frequency_penalty: float= 0.4,
                         **kwargs) -> Any:
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return chat_completion(
                messages, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, **kwargs
            )

        except openai.OpenAIError as e:
            if attempt == MAX_API_RETRIES:
                raise
            wait = BACKOFF_BASE * 2 ** (attempt - 1) * random.uniform(0.8, 1.2)
            logging.warning(
                'API error (%s). Retry %d/%d after %.1fs', e.__class__.__name__, attempt, MAX_API_RETRIES, wait
            )
            time.sleep(wait)


###############################################################################
#  MODEL‑BASED SCORING FUNCTIONS (unchanged)
###############################################################################


def novelty_score(prompt: str) -> float:
    """Higher≈more surprising to the model; simple mean‑logP → perplexity."""
    r = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=0,
        logprobs=1,
    )
    lps = r.choices[0].logprobs.token_logprobs
    return math.exp(-sum(lps) / len(lps))


def answer_difficulty(task: str, answer: str) -> float:
    """Per‑token perplexity of the numeric answer, conditioned on the task."""
    prompt = task.rstrip() + "\n"

    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt + answer,
        temperature=0,
        logprobs=1,
        echo=True,
    )
    lps = resp.choices[0].logprobs.token_logprobs
    ans_tok_count = len(resp.choices[0].logprobs.tokens) - len(
        client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=1,
            logprobs=1,
        ).choices[0].logprobs.tokens
    )
    answer_lps = lps[-ans_tok_count:]
    return math.exp(-sum(answer_lps) / len(answer_lps))

################################################################################
#  EXTRACTION & VALIDATION HELPERS (unchanged)
################################################################################

def extract_blocks(text: str) -> Tuple[str, str, str]:
    m_code = TAG_RX["code"].search(text)
    if not m_code:
        fences = re.findall(r"```python\s*([\s\S]*?)\s*```", text, re.I)
        if fences:
            code = fences[-1].strip()
        else:
            raise ValueError("no Python code block found")
    else:
        code = m_code.group(1).strip()

    m_task = TAG_RX["task"].search(text)
    m_sol  = TAG_RX["solution"].search(text)
    if not (m_task and m_sol):
        raise ValueError("missing <task> or <solution>")

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
                return ""
            seen = True
            return ans_match.group(0)
        return re.sub(r"<answer>.*?</answer>", inner, m.group(0), flags=re.S)
    return re.sub(r"<solution>.*?</solution>", _dedup, text, flags=re.S)


def extract_code_task_solution(text: str) -> Tuple[str, str, str]:
    code_match = re.search(r'<code>.*?```python(.*?)```(?:\s*```)?\s*</code>', text, re.S)
    if code_match:
        code = re.sub(r'^```python\s*', '', code_match.group(1)).strip()
    else:
        fenced_blocks = re.findall(r'```python\s*([\s\S]*?)\s*```', text, re.I)
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
        proc = subprocess.run(['python', '-E', '-I', tmp.name], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[:300])
        return proc.stdout.strip()


def code_matches_solution(code: str, solution: str) -> bool:
    try:
        executed_code_result = run_code(code)
    except Exception as e:
        logging.info('Code execution error: %s', e)
        return False

    if isinstance(executed_code_result, str):
        extracted_code_solution = re.search(r'\b\d+(?:\.\d+)?\b', executed_code_result)
        if not extracted_code_solution:
            return False
        extracted_code_solution = float(extracted_code_solution.group())
    else:
        extracted_code_solution = executed_code_result

    extracted_actual_answer = extract_answer(solution)
    return extracted_code_solution == extracted_actual_answer

################################################################################
#  OPENAI CHAT COMPLETION WRAPPER (unchanged except `n` logic already present)
################################################################################

def chat_completion(messages: List[dict], *, log_probs: bool = False,
                    number_of_generated_outputs: int = 1, **kwargs) -> Any:
    if log_probs:
        kwargs['logprobs'] = True
    kwargs['n'] = max(1, number_of_generated_outputs)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_RESPONSE,
        **kwargs,
    )
    if number_of_generated_outputs == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]

################################################################################
#  TEXT → CODE CONVERSION (unchanged)
################################################################################

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
        logging.error('text2code formatting/extraction failed: %s', exc)
        return None

    if not code_matches_solution(code, solution):
        logging.info('text2code produced non‑matching code.')
        return None
    return code

################################################################################
#  AUGMENTATION LOGIC (major changes start here)
################################################################################

def evaluate_candidate(code: str, task: str, scratch: str, sol: str, step_difficulty_target: int) -> Dict[str, Any]:
    """Evaluate a single candidate augmentation and return a dict with metrics."""
    result: Dict[str, Any] = {
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
    result['scratchpad'] = scratch          # keep for inspection
    result['step_count'] = reasoning_steps(scratch)

    if result['step_count'] < step_difficulty_target + 1:
        result['reason'] = 'too_few_steps'
        return result


    if not correct:
        result['reason'] = REASONS['INCORRECT']
        return result

    # Only compute novelty & difficulty if correct (saves API calls)
    novelty = novelty_score(f'{task}\n{sol}')
    difficulty_val = answer_difficulty(task, sol)
    result['novelty'] = novelty
    result['difficulty'] = difficulty_val

    if abs(novelty) < MIN_NOVELTY:
        result['reason'] = REASONS['LOW_NOVELTY']
    elif abs(difficulty_val) < MIN_DIFFICULTY:
        result['reason'] = REASONS['LOW_DIFFICULTY']
    else:
        result['reason'] = REASONS['ACCEPTED']

    return result


def augment_once(base_code: str, base_task: str, base_sol: str, difficulty: int,
                 max_samples: int = SAMPLE_PER_AUG) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    """Generate up to `max_samples` harder variants and pick the best accepted one.

    Returns (best_candidate_or_None, all_attempted_candidates)
    """
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(difficulty=difficulty)
    user_block = (
        VALID_BLOCK_TEMPLATE
        .replace('$PY',   base_code)
        .replace('$TASK', base_task)
        .replace('$SOL',  base_sol)
        .replace('$ANS',  extract_answer(base_sol).__str__())
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_block},
    ]

    replies: List[str] = safe_chat_completion(messages, number_of_generated_outputs=max_samples)
    attempted: List[Dict[str, Any]] = []

    best_candidate: Dict[str, Any] | None = None
    accepted_candidates: List[Dict[str, Any]] = []

    for raw_reply in replies:
        try:
            raw_reply = dedup_answer_tags(raw_reply)
            raw_reply = enforce_format(raw_reply)
            try:
                new_code, new_task, new_sol, scratch = parse_aug_response(raw_reply)
            except ValueError as e:
                logging.info('Format/extraction error: %s', e)
                continue
        except Exception as e:
            logging.info('Format/extraction error: %s', e)
            continue

        cand = evaluate_candidate(new_code, new_task, new_sol, scratch, step_difficulty_target= difficulty)
        attempted.append(cand)

        if cand['reason'] == REASONS['ACCEPTED']:
            accepted_candidates.append(cand)

        if accepted_candidates:
            best_candidate = max(
                accepted_candidates,
                key=lambda c: (c['novelty'], c['difficulty']),  # sort by novelty, then difficulty
            )
        else:
            best_candidate = None
    return best_candidate, attempted

################################################################################
#  ENFORCE FORMAT (unchanged)
################################################################################

def enforce_format(raw: str, retries: int = 2) -> str:
    from textwrap import dedent
    for _ in range(retries + 1):
        try:
            extract_blocks(raw)
            return raw
        except ValueError:
            fix_prompt = [
                {"role": "system", "content": dedent(f"""
                    Your job is to repair ONLY the formatting of the text the
                    user gives you so that it exactly matches this schema:

                    {VALID_BLOCK_TEMPLATE}

                    Do not change any numbers, wording, or variable names.
                    Return *only* the corrected block.
                """)},
                {"role": "user", "content": raw},
            ]
            raw = safe_chat_completion(fix_prompt)
    raise ValueError("Unable to repair formatting after retries")

################################################################################
#  MAIN DATAFRAME AUGMENTATION LOOP (rewritten)
################################################################################

def augment_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Augment the input DataFrame and return (best_df, all_attempts_df)."""
    best_rows: List[Dict[str, Any]] = []
    all_rows:  List[Dict[str, Any]] = []

    for idx, question, answer in df[['question', 'answer']].itertuples(index=True, name=None):
        task, solution = question, answer
        logging.info('[%s] %s → %s', idx, task[:60], solution)

        base_code = text2code(task, solution)
        if base_code is None:
            logging.info('Failed to obtain base code; skipping.')
            continue

        curr_code, curr_task, curr_sol = base_code, task, solution

        for level in range(1, N_AUGS_PER_SOURCE + 1):
            logging.info('Augmenting level %d', level + 1)
            best_candidate, attempts = augment_once(curr_code, curr_task, curr_sol, difficulty=level + 1)

            # Record every attempt
            for cand in attempts:
                row_dict = {
                    'source_idx': idx,
                    'level': level,
                    **cand,
                }
                all_rows.append(row_dict)

            # If we found an accepted candidate, advance the chain
            if best_candidate is not None:
                best_rows.append({
                    'source_idx': idx,
                    'level': level,
                    **best_candidate,
                })
                curr_code, curr_task, curr_sol = (best_candidate['code'],
                                                   best_candidate['task'],
                                                   best_candidate['solution'])
            else:
                logging.info('No accepted candidate at level %d; stopping.', level + 1)
                break

    return pd.DataFrame(best_rows), pd.DataFrame(all_rows)

################################################################################
#  SCRIPT ENTRY POINT
################################################################################

if __name__ == '__main__':
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main', split='train')
    gsm8k_train = pd.DataFrame(gsm8k_dataset)[:1]

    best_df, all_df = augment_dataframe(gsm8k_train)

    best_df.to_csv('augmented_best.csv', index=False)
    all_df.to_csv('augmented_all.csv', index=False)

    print('Saved augmented_best.csv and augmented_all.csv')
