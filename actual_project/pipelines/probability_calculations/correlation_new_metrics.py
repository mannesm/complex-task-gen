from __future__ import annotations

import asyncio
import logging
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import entropy as voting_entropy, pearsonr, spearmanr

sys.path.extend(
    [
        '/gpfs/home6/mmokkenstorm/tmp/complex_task_gen/',
        '/tmp/pycharm_project_977',
        '/home/mmokkenstorm/tmp/complex_task_gen/actual_project',
    ],
)

import pandas as pd
import torch
import tqdm
from openai import AsyncOpenAI
from pipelines.gsm_evaluation_dataset_creation import create_full_gsm8k_test_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration -------------------------------------------------------------
# ---------------------------------------------------------------------------
MODEL_NAME = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
BASE_URL = 'http://localhost:8000/v1'  # local oai mini‑server
MAX_GEN_TOKENS = 2000  # keep generation short; CoT is long
N_SAMPLES = 5  # k for pass@1
BATCH_SIZE = 32  # GPU batch for log‑prob scoring
ASYNC_LIMIT = 32  # max concurrent chat calls
CHECKPOINT_INTERVAL = 100  # examples per checkpoint write
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

PROMPT_TMPL = """You are a math reasoning assistant.

**Question**: {problem}
{answer}"""  # answer left blank for generation/log‑prob

EVALUATION_PROMPT = """Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?
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


def save_full_dataframe(rows: list[dict], step: int | None = None):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    step_str = f'_step{step}' if step is not None else ''
    path = CHECKPOINT_DIR / f'eval_full{step_str}_{ts}.json'
    pd.DataFrame(rows).to_json(path, orient='records', lines=False, indent=2)
    logger.info('Full checkpoint saved to %s', path)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s', '%Y-%m-%d %H:%M:%S')
stream = logging.StreamHandler()
stream.setFormatter(_formatter)
logger.addHandler(stream)
for noisy in ['httpx', 'urllib3', 'openai', 'transformers']:
    logging.getLogger(noisy).setLevel(logging.WARNING)

regex_pattern_list = [
    r'The answer is\s+\$?\\?\(?boxed\)?\{?([0-9]+(?:\.[0-9]+)?)\}?',  # The answer is \boxed{X} or variants
    r'####\s*([\d,\.]+)',  # #### X
    r'\(\s*\\?boxed\s*\{?\$?([\d,\.]+)\}?\)',  # (\boxed{X}) or similar LaTeX
    r'\$?\\?boxed\s*\{?\$?([\d,\.]+)\}?',  # \boxed{X} without parentheses
    r'The final answer is\s+\$?\\?\(?boxed\)?\{?([0-9]+(?:\.[0-9]+)?)\}?',  # The final answer is boxed{X}
    r'The final answer is\s+\$?([0-9]+(?:\.[0-9]+)?)',  # The final answer is 42
    r'= ([0-9]+(?:\.[0-9]+)?)\s*[\.\)]?\s*$',  # Ends in '= 42.' or '= 42)'
    r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:dollars|pounds|km|miles|liters)?',  # Just a number with optional unit
]


def extract_numeric_value(input_string: str) -> float | None:
    for pattern in regex_pattern_list:
        match = re.search(pattern, input_string)
        if match:
            try:
                return float(match.group(1).replace(',', ''))  # Strip commas for thousands
            except ValueError:
                continue
    return None


class EnhancedLogPScorer:
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='auto')
        self.model.eval()

    @torch.no_grad()
    def score_detailed(self, prompt: str, answer: str) -> dict:
        full_text = prompt + answer
        enc = self.tok(full_text, return_tensors='pt').to(self.model.device)
        plen = len(self.tok(prompt).input_ids)
        labels = enc.input_ids.clone()
        labels[:, :plen] = -100

        outputs = self.model(**enc, labels=labels)
        logits = outputs.logits[0, plen - 1 : -1].float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        token_ids = enc.input_ids[0][plen:].cpu().tolist()
        tokens = self.tok.convert_ids_to_tokens(token_ids)
        lp = log_probs[torch.arange(len(token_ids)), token_ids].tolist()

        entropies = []
        for logit in logits:
            probs = torch.nn.functional.softmax(logit, dim=-1).cpu().numpy()
            entropies.append(-np.sum(probs * np.log(probs + 1e-12)))

        token_details = [
            {'position': i, 'token': t, 'token_id': tid, 'logprob': l, 'prob': math.exp(l)}
            for i, (t, tid, l) in enumerate(zip(tokens, token_ids, lp, strict=False))
        ]

        return {
            'logp_sum': sum(lp),
            'avg_logp': sum(lp) / len(lp),
            'perplexity': math.exp(-sum(lp) / len(lp)),
            'n_tokens': len(lp),
            'token_entropy': float(np.mean(entropies)),
            'prompt': prompt,
            'answer': answer,
            'tokens': token_details,
        }


def compute_voting_entropy(answers: list[str]) -> float:
    if not answers:
        return 0.0
    counts = Counter(answers)
    if len(counts) == 1:
        return 0.0  # No diversity, entropy is 0
    probs = np.array(list(counts.values()), dtype=np.float64)
    probs /= probs.sum()
    return float(voting_entropy(probs))


client = AsyncOpenAI(base_url=BASE_URL, api_key='EMPTY')
sem = asyncio.Semaphore(ASYNC_LIMIT)


async def pass_at_1_async(question: str, numeric_gt: float, k: int = N_SAMPLES) -> tuple[bool, list[str]]:
    if numeric_gt is None:
        return False, []
    formatted_question = EVALUATION_PROMPT.format(question=question)
    messages = [
        {
            'role': 'system',
            'content': 'You are helpful assistant. Solve the question step-by-step. At the end, write: "The final answer is <number>".',
        },
        {'role': 'user', 'content': formatted_question},
    ]
    async with sem:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_GEN_TOKENS,
            temperature=0.7,
            n=k,
        )
    answers = [choice.message.content for choice in resp.choices]
    numeric_preds = [extract_numeric_value(a) for a in answers]
    pass_1 = any(a is not None and abs(a - numeric_gt) < 1e-6 for a in numeric_preds)
    return pass_1, answers


def save_checkpoint(step: int, rows: list[dict]):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = CHECKPOINT_DIR / f'rows_step{step}_{ts}_1_5_b_sub_new_prompt.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info('Checkpoint → %s (%d rows)', path.name, len(rows))


def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    df['pass1'] = df['pass1'].astype(int)
    rows = []
    for col in ['logp_sum', 'avg_logp', 'perplexity', 'token_entropy', 'prompt_len', 'num_ops', 'voting_entropy']:
        pear, p_pear = pearsonr(df[col], df['pass1'])
        spear, p_spear = spearmanr(df[col], df['pass1'])
        rows.append(
            {
                'metric': col,
                'pearson': pear,
                'pearson_p': p_pear,
                'spearman': spear,
                'spearman_p': p_spear,
            },
        )
    return pd.DataFrame(rows)


async def run_eval(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Evaluating %d examples (batch=%d, async_limit=%d)', len(df), BATCH_SIZE, ASYNC_LIMIT)
    scorer = EnhancedLogPScorer(MODEL_NAME)
    rows: list[dict] = []

    for start in tqdm.trange(0, len(df), BATCH_SIZE, desc='batches'):
        chunk = df.iloc[start : start + BATCH_SIZE].copy()
        chunk['prompt_len'] = chunk['question'].apply(lambda x: len(x.split()))
        chunk['num_ops'] = chunk['question'].apply(lambda x: len(re.findall(r'[\+\-\*/]', x)))

        qs = chunk['question'].tolist()
        ans_text = chunk['answer'].tolist()
        nums_gt = [extract_numeric_value(a) for a in ans_text]
        prompts = [PROMPT_TMPL.format(problem=q, answer='') for q in qs]

        pass_tasks = [asyncio.create_task(pass_at_1_async(q, n)) for q, n in zip(qs, nums_gt, strict=False)]
        pass_outputs = await asyncio.gather(*pass_tasks)

        pass_results = []
        generated_answer_lists = []
        for p, g_list in pass_outputs:
            pass_results.append(p)
            generated_answer_lists.append(g_list)

        score_metrics = await asyncio.to_thread(
            lambda: [scorer.score_detailed(p, a) for p, a in zip(prompts, ans_text, strict=False)],
        )

        for row, g_list, passed, prompt, num in zip(
            chunk.itertuples(),
            generated_answer_lists,
            pass_results,
            prompts,
            nums_gt,
            strict=False,
        ):
            # Extract numeric values for each generated answer
            numeric_preds = [extract_numeric_value(a) for a in g_list]

            # Find the first correct answer (if any)
            correct_index = next(
                (i for i, a in enumerate(numeric_preds) if a is not None and abs(a - num) < 1e-6),
                None,
            )
            selected_index = correct_index if correct_index is not None else 0
            selected_answer = g_list[selected_index]
            selected_numeric_answer = extract_numeric_value(selected_answer)

            # Compute log-probability metrics for the selected answer only
            detailed = scorer.score_detailed(prompt, selected_answer)

            rows.append(
                {
                    'id': getattr(row, 'id', row.Index),
                    'difficulty': getattr(row, 'difficulty', 'unk'),
                    'pass1': int(passed),
                    'numeric_answer': num,
                    'generated_answer': selected_answer,
                    'generated_numeric_answer': selected_numeric_answer,
                    'correct_answer': selected_answer if correct_index is not None else None,
                    'generated_answer_list': g_list,
                    'voting_entropy': compute_voting_entropy(g_list),
                    'answer_diversity': len(set(g_list)),
                    'prompt_len': row.prompt_len,
                    'num_ops': row.num_ops,
                    'is_correct_present': correct_index is not None,
                    **{
                        k: detailed[k]
                        for k in [
                            'logp_sum',
                            'avg_logp',
                            'perplexity',
                            'token_entropy',
                            'n_tokens',
                            'prompt',
                            'answer',
                            'tokens',
                        ]
                    },
                },
            )
        if len(rows) and (len(rows) % CHECKPOINT_INTERVAL == 0):
            save_checkpoint(len(rows), rows)

    save_checkpoint(len(rows), rows)
    save_full_dataframe(rows)

    df_out = pd.DataFrame(rows)
    folder_path = '/home/mmokkenstorm/tmp/complex_task_gen/output/'
    df_out.to_json(folder_path + '/gsm8k_pass1_logp_15b_sub_new-prompt.json', index=False)
    # df_eval.to_json(folder_path + '/df_eval_gsm8k_base.json', index=False)
    corr = corr_table(df_out)
    corr.to_csv(folder_path + 'correlation_output.csv')
    corr.to_json(folder_path + 'metric_correlations15b_sub_new_prompt.json', index=False)

    logger.info('Finished • acc@1=%.3f • median ppl=%.1f', df_out['pass1'].mean(), df_out['perplexity'].median())
    logger.info('Correlation table:\n%s', corr.to_string(index=False))
    return df_out


if __name__ == '__main__':
    test_df = create_full_gsm8k_test_dataset(to_df=True)

    async def _main():
        await run_eval(test_df)

    asyncio.run(_main())
