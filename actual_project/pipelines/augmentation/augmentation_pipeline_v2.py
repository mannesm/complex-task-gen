"""
augment_math_dataset.py
"""

from __future__ import annotations
import re
import ast
import tempfile
import subprocess
import math
import pandas as pd

from datasets import load_dataset
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = 'Qwen/Qwen2.5-Coder-7B-Instruct'
BASE_URL = 'http://localhost:8000/v1'
N_AUGS_PER_SOURCE = 10  # a1 … a10
MAX_TOKENS_RESPONSE = 800
TEMPERATURE = 0.2

client = OpenAI(base_url=BASE_URL, api_key='EMPTY')

###############################################################################
#  PROMPT TEMPLATES
###############################################################################
SYSTEM_PROMPT_TEMPLATE = """You are an *Augmenter* of quantitative word-problems.
•  Input: <task>, <solution>, <code> (all mutually consistent).
•  Output: **one** harder variant, formatted EXACTLY:
   <code>```python
   …valid Python…
   ```</code>
   <task>…</task>
   <solution>…#### <answer></solution>
•  Difficulty ladder:
   1 one-step ±                 4 + extra entity
   2 two-step same-op           5 two variables
   3 mix of + – × ÷            ⋯ keep increasing logically
•  Current difficulty target: {difficulty}
•  Never repeat earlier text.  Maximise the novelty score.
"""

TEXT2CODE_SYSTEM_PROMPT = """Convert the <task> and <solution> below into working Python
that converts the question answer pair into python code.
Wrap code in triple-back-ticked Python so downstream parsers can extract it.
Don't add any extra text or comments.
The code should be valid and executable.
"""


RX_CODE_BLOCK = re.compile(r'<code>.*?```python(.*?)```.*?</code>', re.S)
RX_TASK_BLOCK = re.compile(r'<task>(.*?)</task>', re.S)
RX_SOL_BLOCK = re.compile(r'<solution>(.*?)</solution>', re.S)
RX_FINAL_ANSWER = re.compile(r'####\s*([+-]?\d+(?:\.\d+)?)')


def _strip(text: str) -> str:
    return text.strip(' \n')


###############################################################################
#  NOVELTY SCORE USING TOKEN LOGPROBS
###############################################################################
def novelty_score(prompt: str) -> float:
    """Higher≈more surprising to the model; simple mean-logP → perplexity."""
    r = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=0,
        max_tokens=0,
        logprobs=1,
    )
    lps = r.choices[0].logprobs.token_logprobs
    return math.exp(-sum(lps) / len(lps))


###############################################################################
#  PARSING & VALIDATION
###############################################################################
def extract_answer(solution: str) -> float:
    m = RX_FINAL_ANSWER.search(solution)
    if not m:
        raise ValueError('no #### <answer> found')
    return float(m.group(1))


def extract_code_task_solution(text: str) -> tuple[str, str, str]:
    code = _strip(RX_CODE_BLOCK.search(text).group(1))
    task = _strip(RX_TASK_BLOCK.search(text).group(1))
    sol = _strip(RX_SOL_BLOCK.search(text).group(1))
    return code, task, sol


def run_code(code: str) -> float:
    with tempfile.NamedTemporaryFile('w+', suffix='.py') as tmp:
        tmp.write(code)
        tmp.flush()
        proc = subprocess.run(
            ['python', '-E', '-I', tmp.name], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[:300])
        return float(ast.literal_eval(proc.stdout.strip()))


def code_matches_solution(code: str, solution: str) -> bool:
    return run_code(code) == extract_answer(solution)


###############################################################################
#  LLM CALL HELPERS
###############################################################################
def chat_completion(messages, **kwargs):
    return (
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_RESPONSE,
            **kwargs,
        )
        .choices[0]
        .message.content
    )


def text2code(task: str, solution: str) -> str:
    messages = [
        {'role': 'system', 'content': TEXT2CODE_SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': f'<task>{task}</task>\n<solution>{solution}</solution>',
        },
    ]
    resp = chat_completion(messages)
    code, *_ = extract_code_task_solution(
        f'<code>{resp}</code><task></task><solution></solution>'
    )
    if not code_matches_solution(code, solution):
        raise RuntimeError('text2code produced code that fails validation')
    return code


def augment_once(
    code: str, task: str, solution: str, difficulty: int
) -> tuple[str, str, str]:
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(difficulty=difficulty)
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {
            'role': 'user',
            'content': f'<code>```python\n{code}\n```</code>\n'
            f'<task>{task}</task>\n'
            f'<solution>{solution}</solution>',
        },
    ]
    reply = chat_completion(messages)
    new_code, new_task, new_sol = extract_code_task_solution(reply)

    if not code_matches_solution(new_code, new_sol):
        raise RuntimeError('Augmented code & solution mismatch')

    return new_code, new_task, new_sol


###############################################################################
#  MAIN LOOP
###############################################################################
def augment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx in df[['question', 'answer']].itertuples():
        print(f'[{idx}] {idx.question} → {idx.answer}')
        task, solution = idx.question, idx.answer
        base_code = text2code(task, solution)
        curr_code, curr_task, curr_sol = base_code, task, solution
        for i in range(1, N_AUGS_PER_SOURCE + 1):
            try:
                curr_code, curr_task, curr_sol = augment_once(
                    curr_code, curr_task, curr_sol, difficulty=i + 1
                )
                # score = novelty_score(curr_task + '\n' + curr_sol)
            except Exception as exc:
                print(f'[row {idx} iter {i}] ✗ {exc}')
                break
            rows.append(
                {
                    'source_row': idx,
                    'iter': i,
                    'task': curr_task,
                    'solution': curr_sol,
                    'code': curr_code,
                    'answer': extract_answer(curr_sol),
                    'novelty': 'very hard',
                }
            )
    return pd.DataFrame(rows)


###############################################################################
#  USAGE EXAMPLE
###############################################################################
if __name__ == '__main__':
    # raw_df must have columns: task, solution
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main', split='train')
    gsm8k_train = pd.DataFrame(gsm8k_dataset)[:10]
    out_df = augment_dataframe(gsm8k_train)
    out_df.to_csv('augmented_questions.csv', index=False)
    print('Saved augmented_questions.csv')
