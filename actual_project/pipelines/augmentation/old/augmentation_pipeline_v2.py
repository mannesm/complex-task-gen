"""
augment_math_dataset.py
"""
from __future__ import annotations

import random
import time
from typing import Any
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

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

MODEL_NAME = 'Qwen/Qwen2.5-Coder-7B-Instruct'
BASE_URL = 'http://localhost:8000/v1'
N_AUGS_PER_SOURCE = 10
MAX_TOKENS_RESPONSE = 10000
TEMPERATURE = 0.4
MIN_NOVELTY = 5
MAX_API_RETRIES = 3


client = OpenAI(base_url=BASE_URL, api_key='EMPTY')

VALID_BLOCK_TEMPLATE = (
    "<code>```python\n"
    "$PY\n"
    "```</code>\n"
    "<task>$TASK</task>\n"
    "<solution>$SOL #### <answer> $ANS </answer></solution>"
)


difficulty = 0
SYSTEM_PROMPT_TEMPLATE = rf"""
You are an **Augmenter** of quantitative word-problems.

──────────────── INPUT (always mutually consistent) ────────────────
{VALID_BLOCK_TEMPLATE.replace('$PY', '…').replace('$TASK', '…')
                      .replace('$SOL',  '…').replace('$ANS',  '…')}

──────────────── OUTPUT (exactly ONE harder variant) ───────────────
{VALID_BLOCK_TEMPLATE.replace('$PY',   '$NEW_PYTHON_CODE')
                      .replace('$TASK', '$NEW_PROBLEM')
                      .replace('$SOL',  '$NEW_SOLUTION')
                      .replace('$ANS',  '$NUMERIC_ANSWER')}

Rules (read carefully):
1. **Return only the block above** – no extra lines, no commentary.
2. Copy every delimiter literally: `<code>`, triple-back-tick, the word
   *python*, `</code>`, `<task>`, `</task>`, `<solution>`, `</solution>`,
   four `#` characters, the tag `<answer>` and its closing tag.
3. The new problem must be strictly harder (difficulty target = {difficulty}).
4. Do **not** repeat any text from the input except the delimiters.
"""


TEXT2CODE_SYSTEM_PROMPT = """Convert the <task> and <solution> below into working Python
that converts the question-answer pair into Python code.
Wrap the code in <code>...</code> tags and triple-back-ticked Python so downstream parsers can extract it.
Don't add any extra text or comments.
The code should be valid and executable.
"""


RX_CODE_BLOCK = re.compile(r'<code>.*?```python(.*?)```.*?</code>', re.S)
RX_TASK_BLOCK = re.compile(r'<task>(.*?)</task>', re.S)
RX_SOL_BLOCK = re.compile(r'<solution>(.*?)</solution>', re.S)
RX_FINAL_ANSWER = re.compile(r'####\s*([+-]?\d+(?:\.\d+)?)')


def _strip(text: str) -> str:
    return text.strip(' \n')

def safe_chat_completion(messages: list[dict], BACKOFF_BASE=1, **kwargs) -> str:
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return chat_completion(messages, **kwargs)
        except openai.OpenAIError as e:
            if attempt == MAX_API_RETRIES:
                raise                         # bubble up
            wait = BACKOFF_BASE * 2 ** (attempt - 1) * random.uniform(0.8, 1.2)
            logging.warning("API error (%s). Retry %d/%d after %.1fs",
                            e.__class__.__name__, attempt, MAX_API_RETRIES, wait)
            time.sleep(wait)



###############################################################################
#  NOVELTY SCORE USING TOKEN LOGPROBS
###############################################################################
def novelty_score(prompt: str) -> float:
    """Higher≈more surprising to the model; simple mean-logP → perplexity."""
    # TODO: Rewrite to use transformers lib
    r = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=0,
        logprobs=1,
    )
    lps = r.choices[0].logprobs.token_logprobs
    return math.exp(-sum(lps) / len(lps))

def answer_difficulty(task: str, answer: str) -> float:
    """
    Per-token perplexity of the numeric answer, **conditioned on the task**.
    Lower ⇒ easier for the model to predict; higher ⇒ harder.
    """
    prompt = task.rstrip() + "\n"        # ← whatever separator you use

    # Ask the model to *score*, not generate. echo=True gives log-probs
    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt + answer,           # task + answer in one string
        temperature=0,
        logprobs=1,
        echo=True,
    )
    # TODO: Rewrite to use transformers lib
    # logprobs for every prompt token, incl. answer tokens at the end
    lps = resp.choices[0].logprobs.token_logprobs
    # All answer tokens are at the end; we need to know how many:
    ans_tok_count = len(resp.choices[0].logprobs.tokens) - len(
        client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=1,
            logprobs=1,
        ).choices[0].logprobs.tokens
    )
    answer_lps = lps[-ans_tok_count:]

    # −mean-logP  → perplexity
    return math.exp(-sum(answer_lps) / len(answer_lps))


TAG_RX = {
    "code":      re.compile(r"<code>\s*```python\s*(.*?)\s*```.*?</code>", re.S|re.I),
    "task":      re.compile(r"<task>(.*?)</task>",                        re.S|re.I),
    "solution":  re.compile(r"<solution>(.*?)</solution>",                re.S|re.I),
    "answer": re.compile(
        r"(?:<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>"   # alt-1 – tags
        r"|####\s*([+-]?\d+(?:\.\d+)?)\b)"                  # alt-2 – hashes
        , re.I),
}

def extract_blocks(text: str) -> tuple[str, str, str]:
    """
    Extract (code, task, solution) – raise if any block is missing.
    Falls back to raw ```python fences when <code>…</code> is absent.
    """
    m_code = TAG_RX["code"].search(text)
    if not m_code:                                     # fallback
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
    """
    Return the numeric answer found either in:
       • <answer> … </answer>
       • or the last line starting with '#### '
    """
    m = TAG_RX['answer'].search(solution)
    if not m:
        raise ValueError('no numeric answer found')

    num_str = m.group(1) or m.group(2)  # <-- key line
    return float(num_str)

def dedup_answer_tags(text: str) -> str:
    def _dedup(m):
        seen = False
        def inner(ans_match):
            nonlocal seen
            if seen:
                return ""             # drop duplicates
            seen = True
            return ans_match.group(0) # keep the first
        return re.sub(r"<answer>.*?</answer>", inner, m.group(0), flags=re.S)
    return re.sub(r"<solution>.*?</solution>", _dedup, text, flags=re.S)

def extract_code_task_solution(text: str) -> tuple[str, str, str]:
    # Try to extract from <code>...</code>
    code_match = re.search(
        r'<code>.*?```python(.*?)```(?:\s*```)?\s*</code>', text, re.S
    )

    if code_match:
        code = re.sub(r'^```python\s*', '', code_match.group(1)).strip()
    else:
        # Fallback: get last ```python ... ``` block in case <code>...</code> is missing
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
        proc = subprocess.run(
            ['python', '-E', '-I', tmp.name], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[:300])
        return proc.stdout.strip()


def code_matches_solution(code: str, solution: str) -> bool:
    executed_code_result = run_code(code)
    # If extracted_code_result is a string, extract the first number
    # from the string using regex
    if isinstance(executed_code_result, str):
        extracted_code_solution  = re.search(r'\b\d+(\.\d+)?\b', executed_code_result)
        extracted_code_solution = float(extracted_code_solution.group())
    else:
        extracted_code_solution = executed_code_result
    extracted_actual_answer = extract_answer(solution)
    if extracted_code_solution:
        if extracted_code_solution == extracted_actual_answer:
            logging.info('code matches solution: %s, '
                         'extracted_code_solution: %s, '
                         'Actual Code Solution: %s', code, extracted_code_solution, extracted_actual_answer)
            return True
        logging.info('code does not match solution. '
                     'Extracted Solution: %s  \n'
                     'extracted actual Answer: %s', extracted_code_solution, extracted_actual_answer)
    logging.info(f"No extracted code solution found for {code}")

    return False


###############################################################################
#  LLM CALL HELPERS
###############################################################################
def chat_completion(messages, log_probs: bool = False, number_of_generated_outputs: int=1,**kwargs):
    """Call the OpenAI API with the given messages."""
    if log_probs:
        kwargs['logprobs'] = True
    if number_of_generated_outputs <= 1:
        kwargs['n'] = number_of_generated_outputs
    response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_RESPONSE,
            **kwargs,
        )
    if number_of_generated_outputs > 1:
        return [choice.message.content for choice in response.choices]
    else:
        return response.choices[0].message.content


def text2code(task: str, solution: str) -> tuple[str | None, Any] | None:
    messages = [
        {'role': 'system', 'content': TEXT2CODE_SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': f'<task>{task}</task>\n<solution>{solution}</solution>',
        },
    ]
    resp = safe_chat_completion(messages)
    resp = enforce_format(resp)  
    code, *_ = extract_blocks(resp)


    logging.info('text2code response: %s', resp)
    try:
        if not resp.strip().startswith('<code>'):
            resp = f'<code>\n{resp}\n</code>'
        else:
            resp = re.sub(r'```python\s*```', '```python', resp)
        logging.info('extracting code from response')
        code, *_ = extract_code_task_solution(
            f'{resp}<task></task><solution></solution>'
        )
        logging.info('extracted code: %s', code)
    except Exception as exc:
        logging.error(f'failed to extract code: {exc}')
        return None
    try:
        if not code_matches_solution(code, solution):
            raise RuntimeError('text2code produced code that fails validation')
    except Exception:
        return None
    return resp


def augment_once(code: str, task: str, solution: str, difficulty: int,
                 tries: int = 5) -> tuple[str, str, str, bool] | None:

    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(difficulty=difficulty)
    user_block = (VALID_BLOCK_TEMPLATE
                  .replace('$PY',   code)
                  .replace('$TASK', task)
                  .replace('$SOL',  solution)
                  .replace('$ANS',  extract_answer(solution).__str__()))

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_block},
    ]
    is_correct = False                           # default

    for attempt in range(1, tries + 1):
        reply = safe_chat_completion(messages)
        try:
            dedup_answer_tags(reply)
        except Exception as exc:
            logging.error(f'failed to dedup answer tags: {exc}')
        reply = enforce_format(reply)
        new_code, new_task, new_sol = extract_blocks(reply)

        if code_matches_solution(new_code, new_sol):
            is_correct = True
            logging.info('code matches solution: %s', new_code)
            return new_code, new_task, new_sol, is_correct

        logging.info("Validation failed (%d/%d). Retrying …", attempt, tries)
        return new_code, new_task, new_sol, is_correct



def enforce_format(raw: str, retries: int = 2) -> str:
    """
    Guarantee that `raw` conforms to VALID_BLOCK_TEMPLATE.
    If it does not, ask the model to *repair only the delimiters*
    (up to `retries` attempts).  Raises on final failure.
    """
    from textwrap import dedent   # local import to avoid clutter at top

    for _ in range(retries + 1):
        try:
            extract_blocks(raw)      # ← will raise ValueError if malformed
            return raw
        except ValueError:
            fix_prompt = [
                {"role": "system", "content": dedent(f"""\
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


def augment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx in df[['question', 'answer']].itertuples():
        # limit the logging of the string to 50 characters
        logging.info(f'[{idx}] {idx.question[:50]} → {idx.answer[:50]}')
        task, solution = idx.question, idx.answer
        base_code = text2code(task, solution)
        logging.info(f'base code: {base_code}')
        if not base_code:  # plainly failed
            logging.info(f'no code for task {task}')
            continue  # skip this source row

        curr_code, curr_task, curr_sol = base_code, task, solution
        for i in range(1, N_AUGS_PER_SOURCE + 1):
            last_good = (curr_code, curr_task, curr_sol, True)
            try:
                logging.info("Augmenting with difficulty %d", i + 1)
                curr_code, curr_task, curr_sol, ok = augment_once(
                    curr_code, curr_task, curr_sol, difficulty=i + 1
                )
                logging.info('Augmented code: %s', curr_code)
                logging.info('Augmented task: %s', curr_task)
                logging.info('Augmented solution: %s', curr_sol)
                novelty_rating = novelty_score(f'{curr_task}\n{curr_sol}')
                logging.info('Novelty rating: %s', novelty_rating)
                difficulty_score = answer_difficulty(curr_task, curr_sol)
                logging.info('Difficulty score: %s', difficulty_score)
            except Exception as exc:
                logging.info(f'error failed to augment: {exc}')
                curr_code, curr_task, curr_sol, ok = last_good
                ok = False
                break
            if rows and rows[-1]['task'] == curr_task:
                logging.info('Already augmented')
                continue
            rows.append(
                {
                    'source_row': idx,
                    'iter': i,
                    'task': curr_task,
                    'solution': curr_sol,
                    'code': curr_code,
                    'novelty': novelty_rating,
                    'difficulty': difficulty_score,
                    'is_correct': ok
                }
            )
    return pd.DataFrame(rows)


if __name__ == '__main__':
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main', split='train')
    gsm8k_train = pd.DataFrame(gsm8k_dataset)[:1]
    out_df = augment_dataframe(gsm8k_train)
    out_df.to_csv('augmented_questions_300_1000.csv', index=False)
    print('Saved augmented_questions_.csv')