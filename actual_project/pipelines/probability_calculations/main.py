# main.py
import json
import logging
import math
import re

import pandas as pd
import torch
import tqdm
from openai import OpenAI
from scipy.stats import pearsonr, spearmanr

from complex_task_gen.actual_project.pipelines.gsm_evaluation_dataset_creation import (
    create_gsm_evaluation_datasets,
)

# Import LogPScorer from logp_utils instead of redefining it
from complex_task_gen.actual_project.pipelines.probability_calculations.logp_utils import LogPScorer

MODEL_NAME = 'Qwen/Qwen2.5-Math-7B-Instruct'
BASE_URL = 'http://localhost:8000/v1'
MAX_GEN_TOKENS = 1024  # keep generation short; CoT is long
N_SAMPLES = 4  # k for pass@1

client = OpenAI(base_url=BASE_URL, api_key='EMPTY')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')

PROMPT = """You are a math reasoning assistant.

**Question**: {problem}
{answer}"""  # simple template; answer left blank for generation/log-P

# ---------- helpers ----------------------------------------------------------


def extract_numeric_from_solution(sol: str) -> float | None:
    """Return the '#### 42' number, or None."""
    m = re.search(r'####\s*([0-9\.,\-]+)', sol)
    if not m:
        return None
    try:
        return float(m.group(1).replace(',', ''))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Extend LogPScorer to capture more detailed information (stable fp32 math)
# --------------------------------------------------------------------------
class EnhancedLogPScorer(LogPScorer):
    @torch.no_grad()
    def score_detailed(self, prompt: str, answer: str):
        full = prompt + answer

        # ----- encode once --------------------------------------------------
        enc = self.tok(full, return_tensors='pt').to(self.model.device)
        plen = len(self.tok(prompt).input_ids)

        # labels: mask the prompt tokens
        labels = enc.input_ids.clone()
        labels[:, :plen] = -100  # -100 ➜ ignored by CE loss

        # ===================================================================
        # disable autocast for the entire forward + log-softmax path
        # ===================================================================
        with torch.autocast(self.model.device.type, enabled=False):
            outputs = self.model(**enc, labels=labels)  # forward pass

            # slice the answer logits and cast **just that slice** to fp32
            answer_logits = outputs.logits[0, plen - 1 : -1].float()
            log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
        # -------------------------------------------------------------------

        # token-level bookkeeping -------------------------------------------
        token_ids = enc.input_ids[0][plen:].cpu()
        tokens = self.tok.convert_ids_to_tokens(token_ids)
        actual_logprobs = log_probs[torch.arange(len(token_ids)), token_ids].tolist()

        # summary metrics ----------------------------------------------------
        logp_sum = sum(actual_logprobs)
        avg_logp = logp_sum / len(actual_logprobs)
        perplexity = math.exp(-avg_logp)
        n_tgt = len(token_ids)

        token_details = [
            {
                'position': i,
                'token': tok,
                'token_id': tid,
                'logprob': lp,
                'prob': math.exp(lp),
            }
            for i, (tok, tid, lp) in enumerate(zip(tokens, token_ids, actual_logprobs, strict=False))
        ]

        return {
            'basic': {
                'logp_sum': logp_sum,
                'avg_logp': avg_logp,
                'perplexity': perplexity,
                'n_tokens': n_tgt,
            },
            'detailed': {
                'tokens': token_details,
                'full_text': full,
                'prompt_length': plen,
                'answer_length': n_tgt,
            },
        }


def pass_at_1(question: str, numeric_gt: float, k=N_SAMPLES) -> bool:
    prompt = PROMPT.format(problem=question, answer='')
    answers = []
    for _ in range(k):
        messages = [
            {'role': 'system', 'content': 'You are a math-reasoning assistant.'},
            {'role': 'user', 'content': question},
        ]
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_GEN_TOKENS,
            temperature=0.7,
        )
        answers.append(resp.choices[0].message.content)
        logging.info(answers)
    nums = [extract_numeric_from_solution(a) for a in answers]
    return any(a is not None and abs(a - numeric_gt) < 1e-6 for a in nums)


# ---------- main loop --------------------------------------------------------


def run_eval(df: pd.DataFrame) -> pd.DataFrame:
    # df = test_df[:2]
    scorer = EnhancedLogPScorer(MODEL_NAME)
    rows = []
    detailed_results = []

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        q, sol_text = row['question'], row['answer']
        num_ans = extract_numeric_from_solution(sol_text)
        if num_ans is None:
            logger.warning('No numeric answer for row %s – skipped', i)
            continue

        prompt = PROMPT.format(problem=q, answer='')

        # Get detailed scoring information
        result = scorer.score_detailed(prompt, sol_text)
        basic_metrics = result['basic']

        # Save the detailed results separately (JSON format is better for nested data)
        detailed_result = {
            'id': row.get('id', i),
            'question': q,
            'answer': sol_text,
            'numeric_answer': num_ans,
            'detailed_metrics': result,
        }
        detailed_results.append(detailed_result)

        # Check if the model passes@1
        passed = pass_at_1(q, num_ans, k=N_SAMPLES)

        # Add summary metrics to the DataFrame
        rows.append(
            dict(
                id=row.get('id', i),
                difficulty=row.get('difficulty', 'unk'),
                logp_sum=basic_metrics['logp_sum'],
                avg_logp=basic_metrics['avg_logp'],
                perplexity=basic_metrics['perplexity'],
                n_tokens=basic_metrics['n_tokens'],
                pass1=int(passed),
            ),
        )

    with open('../../../output/gsm8k_detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)

    return pd.DataFrame(rows)


def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    df['pass1'] = df['pass1'].astype(int)
    out = []
    for col in ['logp_sum', 'avg_logp', 'perplexity']:
        pear, p1 = pearsonr(df[col], df['pass1'])
        spear, p2 = spearmanr(df[col], df['pass1'])
        out.append(dict(metric=col, pearson=pear, pearson_p=p1, spearman=spear, spearman_p=p2))
    return pd.DataFrame(out)


if __name__ == '__main__':
    base, easy, med, hard = create_gsm_evaluation_datasets(to_df=True)
    test_df = pd.concat(
        [
            base.assign(difficulty='base'),
            easy.assign(difficulty='easy'),
            med.assign(difficulty='medium'),
            hard.assign(difficulty='hard'),
        ],
    ).reset_index(drop=True)

    res = run_eval(test_df)
    res.to_csv('gsm8k_pass1_logp.csv', index=False)

    corr = corr_table(res)
    corr.to_csv('metric_correlations.csv', index=False)
    logger.info('\n' + corr.to_string(index=False))
