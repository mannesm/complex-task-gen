from __future__ import annotations

import asyncio
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.extend(
    [
        '/tmp/R9o8sQTVOL/',
        '/tmp/pycharm_project_977',
    ],
)

import pandas as pd
import torch
import tqdm
from openai import AsyncOpenAI
from pipelines.gsm_evaluation_dataset_creation import create_full_gsm8k_test_dataset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration -------------------------------------------------------------
# ---------------------------------------------------------------------------
MODEL_NAME = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
BASE_URL = 'http://localhost:8000/v1'  # local oai mini‑server
MAX_GEN_TOKENS = 512  # keep generation short; CoT is long
N_SAMPLES = 1  # k for pass@1
BATCH_SIZE = 16  # GPU batch for log‑prob scoring
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


ANS_RE = re.compile(r'(?:####\s*|The answer is\s*|boxed\s{\s*)([0-9\.,\-]+)(?:\s*})?')

# ---------------------------------------------------------------------------
# Logging -------------------------------------------------------------------
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s', '%Y-%m-%d %H:%M:%S')
stream = logging.StreamHandler()
stream.setFormatter(_formatter)
logger.addHandler(stream)

for noisy in ['httpx', 'urllib3', 'openai', 'transformers']:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

import numpy as np


def sequence_entropy(token_probs: list[list[float]]) -> float:
    return -np.mean([np.sum(p * np.log(p + 1e-12)) for p in token_probs])


def extract_numeric_from_solution(sol: str) -> float | None:
    """Return the numeric part of a GSM8K '#### 42' answer, or *None*."""
    m = ANS_RE.search(sol)
    if not m:
        return None
    try:
        return float(m.group(1).replace(',', ''))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Log‑P scorer with batching -------------------------------------------------
# ---------------------------------------------------------------------------
class BatchedLogPScorer:
    """GPU‑efficient log‑prob scorer (fp16) that handles *lists* of examples."""

    def __init__(self, model_name: str, half: bool = True):
        dtype = torch.float16 if half else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map='auto',
        )
        self.model.eval()

    @torch.no_grad()
    def score_batch(
        self,
        prompts: list[str],
        answers: list[str],
    ) -> list[dict]:
        """Return basic metrics for *each* pair (*logp_sum*, *avg_logp*, *ppl*, *n_tokens*)."""
        full_txt = [p + a for p, a in zip(prompts, answers, strict=False)]
        enc = self.tok(
            full_txt,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(self.model.device)

        prompt_lens = [len(self.tok(p).input_ids) for p in prompts]
        labels = enc.input_ids.clone()
        for row, plen in enumerate(prompt_lens):
            labels[row, :plen] = -100  # mask prompt

        outputs = self.model(**enc, labels=labels)
        logits = outputs.logits  # (B, L, vocab)

        metrics: list[dict] = []
        for row, plen in enumerate(prompt_lens):
            tgt_mask = labels[row] != -100
            l_gt = labels[row][tgt_mask]
            l_pred = logits[row, tgt_mask.nonzero(as_tuple=True)[0] - 1]
            log_probs = torch.log_softmax(l_pred.float(), dim=-1)
            chosen = log_probs[torch.arange(len(l_gt)), l_gt]
            logp_sum = chosen.sum().item()
            avg_logp = chosen.mean().item()
            perplexity = math.exp(-avg_logp)
            logging.info(
                'logp_sum: %s',
                logp_sum,
                'avg_logp: %s',
                avg_logp,
                'perplexity: %s',
                perplexity,
                'prompt: %s',
                prompts[row][:20],
            )
            metrics.append(
                {
                    'logp_sum': logp_sum,
                    'avg_logp': avg_logp,
                    'perplexity': perplexity,
                    'n_tokens': len(l_gt),
                },
            )
        return metrics


# ---------------------------------------------------------------------------
# Asynchronous pass@1 --------------------------------------------------------
# ---------------------------------------------------------------------------
client = AsyncOpenAI(base_url=BASE_URL, api_key='EMPTY')
sem = asyncio.Semaphore(ASYNC_LIMIT)


async def pass_at_1_async(question: str, numeric_gt: float, k: int = N_SAMPLES) -> tuple[bool, str]:
    """Return a tuple:
    - *True* if any of *k* completions match *numeric_gt*
    - The first generated response (or full list if you prefer)
    """
    if numeric_gt is None:
        return False, ''

    formatted_question = EVALUATION_PROMPT.format(question=question)
    messages = [
        {
            'role': 'system',
            'content': 'You are helpful assistant. Always answer the question in numeric. End with "the answer is <number>."',
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
    numeric_preds = [extract_numeric_from_solution(a) for a in answers]

    pass_1 = any(a is not None and abs(a - numeric_gt) < 1e-6 for a in numeric_preds)
    return pass_1, answers  # you can also return the whole list if needed


# ---------------------------------------------------------------------------
# Checkpointing --------------------------------------------------------------
# ---------------------------------------------------------------------------
class EnhancedLogPScorer:
    """Provides per-token log-probs and metadata for each (prompt, answer) pair."""

    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # precise token logprobs
            device_map='auto',
        )
        self.model.eval()

    @torch.no_grad()
    def score_detailed(self, prompt: str, answer: str) -> dict:
        full_text = prompt + answer
        enc = self.tok(full_text, return_tensors='pt').to(self.model.device)
        plen = len(self.tok(prompt).input_ids)

        labels = enc.input_ids.clone()
        labels[:, :plen] = -100  # ignore prompt in loss

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
            {
                'position': i,
                'token': t,
                'token_id': tid,
                'logprob': l,
                'prob': math.exp(l),
            }
            for i, (t, tid, l) in enumerate(zip(tokens, token_ids, lp, strict=False))
        ]

        return {
            'logp_sum': sum(lp),
            'avg_logp': sum(lp) / len(lp),
            'perplexity': math.exp(-sum(lp) / len(lp)),
            'n_tokens': len(lp),
            'prompt': prompt,
            'answer': answer,
            'tokens': token_details,
        }


def save_checkpoint(step: int, rows: list[dict]):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = CHECKPOINT_DIR / f'rows_step{step}_{ts}_1_5_b.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info('Checkpoint → %s (%d rows)', path.name, len(rows))


# ---------------------------------------------------------------------------
# Correlation helper ---------------------------------------------------------
# ---------------------------------------------------------------------------


def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    df['pass1'] = df['pass1'].astype(int)
    rows = []
    for col in ['logp_sum', 'avg_logp', 'perplexity']:
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


# ---------------------------------------------------------------------------
# Main evaluation loop -------------------------------------------------------
# ---------------------------------------------------------------------------
async def run_eval(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Evaluating %d examples (batch=%d, async_limit=%d)', len(df), BATCH_SIZE, ASYNC_LIMIT)

    scorer = EnhancedLogPScorer(MODEL_NAME)
    rows: list[dict] = []

    # iterate over chunks for GPU scoring
    for start in tqdm.trange(0, len(df), BATCH_SIZE, desc='batches'):
        chunk = df.iloc[start : start + BATCH_SIZE]

        qs = chunk['question'].tolist()
        ans_text = chunk['answer'].tolist()
        nums_gt = [extract_numeric_from_solution(a) for a in ans_text]

        prompts = [PROMPT_TMPL.format(problem=q, answer='') for q in qs]

        # ---- launch pass@1 tasks *first* so they run while GPU is busy ------
        pass_tasks = [asyncio.create_task(pass_at_1_async(q, n)) for q, n in zip(qs, nums_gt, strict=False)]
        pass_outputs = await asyncio.gather(*pass_tasks)

        pass_results = []
        generated_answer_lists = []
        for p, g_list in pass_outputs:
            pass_results.append(p)
            generated_answer_lists.append(g_list)
        from collections import Counter

        from scipy.stats import entropy as voting_entropy

        def compute_voting_entropy(answers: list[str]) -> float:
            counts = Counter(answers)
            probs = np.array(list(counts.values())) / sum(counts.values())
            return float(voting_entropy(probs))

        # score batch on dedicated CPU thread so the event loop can progress
        score_metrics = await asyncio.to_thread(
            lambda: [scorer.score_detailed(p, a) for p, a in zip(prompts, ans_text, strict=False)],
        )

        # aggregate ---------------------------------------------------------
        for row, detailed, passed, gen_ans, num in zip(
            chunk.itertuples(),
            score_metrics,
            pass_results,
            generated_answers,
            nums_gt,
            strict=False,
        ):
            rows.append(
                {
                    'id': getattr(row, 'id', row.Index),
                    'difficulty': getattr(row, 'difficulty', 'unk'),
                    'pass1': int(passed),
                    'numeric_answer': num,
                    'generated_answer': gen_ans,
                    **{k: detailed[k] for k in ['logp_sum', 'avg_logp', 'perplexity', 'n_tokens']},
                    'prompt': detailed['prompt'],
                    'answer': detailed['answer'],
                    'tokens': detailed['tokens'],
                    'generated_numeric_answer': extract_numeric_from_solution(gen_ans),
                },
            )

        # checkpoint every N examples --------------------------------------
        if len(rows) and (len(rows) % CHECKPOINT_INTERVAL == 0):
            save_checkpoint(len(rows), rows)

    # final outputs ---------------------------------------------------------
    save_checkpoint(len(rows), rows)
    save_full_dataframe(rows)

    df_out = pd.DataFrame(rows)
    df_out.to_json('gsm8k_pass1_logp_15b.json', index=False)
    corr = corr_table(df_out)
    corr.to_json('metric_correlations15b.json', index=False)

    logger.info('Finished • acc@1=%.3f • median ppl=%.1f', df_out['pass1'].mean(), df_out['perplexity'].median())
    logger.info('Correlation table:\n%s', corr.to_string(index=False))
    return df_out


# ---------------------------------------------------------------------------
# CLI entrypoint -------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Small demo run with 100 examples; change slice as needed
    test_df = create_full_gsm8k_test_dataset(to_df=True)

    async def _main():
        await run_eval(test_df)  # evaluate entire set

    asyncio.run(_main())
