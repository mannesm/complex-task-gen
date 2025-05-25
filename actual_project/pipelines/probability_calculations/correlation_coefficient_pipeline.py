import json
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
from openai import OpenAI
from pipelines.gsm_evaluation_dataset_creation import (
    create_full_gsm8k_test_dataset,
)
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'Qwen/Qwen2.5-Math-7B-Instruct'
BASE_URL = 'http://localhost:8000/v1'
MAX_GEN_TOKENS = 1024  # keep generation short; CoT is long
N_SAMPLES = 4  # k for pass@1
CHECKPOINT_INTERVAL = 10  # save results every N examples
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

PROMPT = """You are a math reasoning assistant.

**Question**: {problem}
{answer}"""  # simple template; answer left blank for generation/log-P

# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_formatter = logging.Formatter(
    '%(asctime)s  %(levelname)s  %(name)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
_console = logging.StreamHandler()
_console.setFormatter(_formatter)
_file = logging.FileHandler('run_eval.log', mode='w')
_file.setFormatter(_formatter)
logger.addHandler(_console)
logger.addHandler(_file)

# Calm down excessively chatty libraries
for noisy in ['httpx', 'urllib3', 'openai', 'transformers']:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# OpenAI client --------------------------------------------------------------
# ---------------------------------------------------------------------------
client = OpenAI(base_url=BASE_URL, api_key='EMPTY')

# ---------------------------------------------------------------------------
# Helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------


def extract_numeric_from_solution(sol: str) -> float | None:
    """Return the numeric part of a GSM8K '#### 42' answer, or *None* if missing."""
    m = re.search(r'####\s*([0-9\.,\-]+)', sol)
    if not m:
        return None
    try:
        return float(m.group(1).replace(',', ''))
    except ValueError:
        return None


class LogPScorer:
    def __init__(self, model_name: str, half=True):
        dtype = torch.float16
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map='auto')
        self.model.eval()

    @torch.no_grad()
    def score_answer(self, prompt: str, answer: str):
        """Log p(answer | prompt).  Returns (logp_sum, avg_logp, perplexity)."""
        full = prompt + answer  # single sequence, causal conditioning
        enc = self.tok(full, return_tensors='pt').to(self.model.device)
        plen = len(self.tok(prompt).input_ids)

        # create labels masking out the prompt
        labels = enc.input_ids.clone()
        labels[:, :plen] = -100  # -100 ➜ ignored by CE loss

        out = self.model(**enc, labels=labels)
        # cross-entropy is returned *averaged over unmasked positions*
        loss = out.loss
        n_tgt = (labels != -100).sum().item()

        avg_logp = -loss.item()  # negative CE
        logp_sum = avg_logp * n_tgt
        perplexity = math.exp(-avg_logp)
        return logp_sum, avg_logp, perplexity, n_tgt


# ---------------------------------------------------------------------------
# Log‑P scorer with token‑level details --------------------------------------
# ---------------------------------------------------------------------------
class EnhancedLogPScorer(LogPScorer):
    """Extends *LogPScorer* to emit fine‑grained token statistics in pure fp32."""

    @torch.no_grad()
    def score_detailed(self, prompt: str, answer: str):
        full = prompt + answer

        enc = self.tok(full, return_tensors='pt').to(self.model.device)
        plen = len(self.tok(prompt).input_ids)

        # Mask prompt tokens
        labels = enc.input_ids.clone()
        labels[:, :plen] = -100  # -100 ➜ ignored by CE loss

        with torch.autocast(self.model.device.type, enabled=False):
            outputs = self.model(**enc, labels=labels)
            answer_logits = outputs.logits[0, plen - 1 : -1].float()
            log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
            token_ids = enc.input_ids[0][plen:].cpu().tolist()
        tokens = self.tok.convert_ids_to_tokens(token_ids)
        lp = log_probs[torch.arange(len(token_ids)), token_ids].tolist()

        logp_sum = sum(lp)
        avg_logp = logp_sum / len(lp)
        perplexity = math.exp(-avg_logp)

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
            'basic': {
                'logp_sum': logp_sum,
                'avg_logp': avg_logp,
                'perplexity': perplexity,
                'n_tokens': len(lp),
            },
            'detailed': {
                'tokens': token_details,
                'full_text': full,
                'prompt_length': plen,
                'answer_length': len(lp),
            },
        }


def save_checkpoint(step: int, rows: list[dict], details: list[dict]) -> None:
    """Serialize interim results every *CHECKPOINT_INTERVAL* steps."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rows_file = CHECKPOINT_DIR / f'rows_step{step}_{ts}.csv'
    details_file = CHECKPOINT_DIR / f'details_step{step}_{ts}.json'

    pd.DataFrame(rows).to_csv(rows_file, index=False)
    with open(details_file, 'w') as fp:
        json.dump(details, fp, indent=2)

    logger.info('Checkpoint → %s • %s', rows_file.name, details_file.name)


def pass_at_1(question: str, numeric_gt: float, k: int = N_SAMPLES) -> bool:
    """Return *True* if **any** of *k* sampled completions match *numeric_gt*.

    A single ChatCompletion request with `n=k` is cheaper and faster than looping.
    """
    messages = [
        {'role': 'system', 'content': 'You are a math‑reasoning assistant.'},
        {'role': 'user', 'content': question},
    ]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_GEN_TOKENS,
        temperature=0.7,
        n=k,
    )

    answers = [choice.message.content for choice in resp.choices]
    logger.debug('Generated answers: %s', answers)

    nums = [extract_numeric_from_solution(a) for a in answers]
    return any(a is not None and abs(a - numeric_gt) < 1e-6 for a in nums)


def run_eval(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Evaluating %d examples', len(df))

    scorer = EnhancedLogPScorer(MODEL_NAME)
    rows: list[dict] = []
    detailed: list[dict] = []

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='scoring'):
        q, sol_text = row['question'], row['answer']
        num_ans = extract_numeric_from_solution(sol_text)
        if num_ans is None:
            logger.warning('No numeric answer for row %s – skipped', i)
            continue

        prompt = PROMPT.format(problem=q, answer='')

        metrics = scorer.score_detailed(prompt, sol_text)
        basic = metrics['basic']

        detailed.append(
            {
                'id': row.get('id', i),
                'question': q,
                'answer': sol_text,
                'numeric_answer': num_ans,
                'metrics': metrics,
            },
        )

        passed = pass_at_1(q, num_ans, k=N_SAMPLES)

        rows.append(
            {
                'id': row.get('id', i),
                'difficulty': row.get('difficulty', 'unk'),
                'logp_sum': basic['logp_sum'],
                'avg_logp': basic['avg_logp'],
                'perplexity': basic['perplexity'],
                'n_tokens': basic['n_tokens'],
                'pass1': int(passed),
            },
        )

        # Periodic checkpoint
        if (len(rows) % CHECKPOINT_INTERVAL) == 0:
            save_checkpoint(len(rows), rows, detailed)

    # Final dump ------------------------------------------------------------
    save_checkpoint(len(rows), rows, detailed)

    # Canonical artefacts for downstream scripts ---------------------------
    with open('../../../output/gsm8k_detailed_results.json', 'w') as fp:
        json.dump(detailed, fp, indent=2)

    df_out = pd.DataFrame(rows)
    df_out.to_csv('gsm8k_pass1_logp.csv', index=False)
    logger.info('Wrote gsm8k_pass1_logp.csv and gsm8k_detailed_results.json')

    return df_out


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
# Entry point ---------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    test_df = create_full_gsm8k_test_dataset(to_df=True)

    results_df = run_eval(test_df[:1])

    corr = corr_table(results_df)
    corr.to_csv('metric_correlations.csv', index=False)
    logger.info('\n%s', corr.to_string(index=False))
