import argparse
import concurrent.futures
import datetime
import json
import logging
import math
import os
import re
import threading
from datetime import datetime

import pandas as pd
import torch
import tqdm
from openai import OpenAI
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from complex_task_gen.actual_project.pipelines.gsm_evaluation_dataset_creation import (
    create_gsm_evaluation_datasets,
)

MODEL_NAME = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
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


class LogPScorer:
    def __init__(self, model_name: str, half=True):
        dtype = torch.float16 if half else torch.float32
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


local_storage = threading.local()


def get_model_and_tokenizer(model_name, half=True):
    """Get or create model and tokenizer for the current thread"""
    if not hasattr(local_storage, 'model') or not hasattr(local_storage, 'tokenizer'):
        logger.info(f'Creating new model instance for thread {threading.current_thread().name}')
        dtype = torch.float16 if half else torch.float32
        local_storage.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        local_storage.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map='auto')
        local_storage.model.eval()
    return local_storage.model, local_storage.tokenizer


def extract_numeric_from_solution(sol: str) -> float | None:
    """Return the '#### 42' number, or None."""
    m = re.search(r'####\s*([0-9\.,\-]+)', sol)
    if not m:
        return None
    try:
        return float(m.group(1).replace(',', ''))
    except ValueError:
        return None


class LogPScorer:
    @torch.no_grad()
    def score_answer(self, prompt: str, answer: str, model_name=MODEL_NAME):
        """Log p(answer | prompt).  Returns (logp_sum, avg_logp, perplexity)."""
        model, tok = get_model_and_tokenizer(model_name)

        full = prompt + answer  # single sequence, causal conditioning
        enc = tok(full, return_tensors='pt').to(model.device)
        plen = len(tok(prompt).input_ids)

        # create labels masking out the prompt
        labels = enc.input_ids.clone()
        labels[:, :plen] = -100  # -100 ➜ ignored by CE loss

        out = model(**enc, labels=labels)
        # cross-entropy is returned *averaged over unmasked positions*
        loss = out.loss
        n_tgt = (labels != -100).sum().item()

        avg_logp = -loss.item()  # negative CE
        logp_sum = avg_logp * n_tgt
        perplexity = math.exp(-avg_logp)
        return logp_sum, avg_logp, perplexity, n_tgt

    @torch.no_grad()
    def score_detailed(self, prompt: str, answer: str, model_name=MODEL_NAME):
        model, tok = get_model_and_tokenizer(model_name)

        full = prompt + answer

        # ----- encode once --------------------------------------------------
        enc = tok(full, return_tensors='pt').to(model.device)
        plen = len(tok(prompt).input_ids)

        # labels: mask the prompt tokens
        labels = enc.input_ids.clone()
        labels[:, :plen] = -100  # -100 ➜ ignored by CE loss

        # ===================================================================
        # disable autocast for the entire forward + log-softmax path
        # ===================================================================
        with torch.autocast(model.device.type, enabled=False):
            outputs = model(**enc, labels=labels)  # forward pass

            # slice the answer logits and cast **just that slice** to fp32
            answer_logits = outputs.logits[0, plen - 1 : -1].float()
            log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
        # -------------------------------------------------------------------

        # token-level bookkeeping -------------------------------------------
        token_ids = enc.input_ids[0][plen:].cpu()
        tokens = tok.convert_ids_to_tokens(token_ids)
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


def process_row(row_data):
    """Process a single row (for parallel execution)"""
    i, row = row_data
    q, sol_text = row['question'], row['answer']
    num_ans = extract_numeric_from_solution(sol_text)
    if num_ans is None:
        logger.warning('No numeric answer for row %s – skipped', i)
        return None

    # Use the shared scorer via thread-local storage
    scorer = LogPScorer()
    thread_client = OpenAI(base_url=BASE_URL, api_key='EMPTY')

    prompt = PROMPT.format(problem=q, answer='')
    result = scorer.score_detailed(prompt, sol_text)
    basic_metrics = result['basic']

    # Pass@1 with thread-local client
    passed = pass_at_1_local(q, num_ans, k=N_SAMPLES, client=thread_client)

    detailed_result = {
        'id': row.get('id', i),
        'question': q,
        'answer': sol_text,
        'numeric_answer': num_ans,
        'detailed_metrics': result,
    }

    summary = dict(
        id=row.get('id', i),
        difficulty=row.get('difficulty', 'unk'),
        logp_sum=basic_metrics['logp_sum'],
        avg_logp=basic_metrics['avg_logp'],
        perplexity=basic_metrics['perplexity'],
        n_tokens=basic_metrics['n_tokens'],
        pass1=int(passed),
    )

    return {'summary': summary, 'detailed': detailed_result}


def pass_at_1_local(question, numeric_gt, k=N_SAMPLES, client=None):
    """Thread-safe version of pass_at_1"""
    if client is None:
        client = OpenAI(base_url=BASE_URL, api_key='EMPTY')

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

    nums = [extract_numeric_from_solution(a) for a in answers]
    return any(a is not None and abs(a - numeric_gt) < 1e-6 for a in nums)


def save_checkpoint(results_dict, detailed_results, checkpoint_file='checkpoint.json'):
    """Save progress to checkpoint file"""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'summaries': results_dict,
        'detailed_results': detailed_results,
    }

    temp_file = f'{checkpoint_file}.temp'
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_data, f)

    os.replace(temp_file, checkpoint_file)
    logger.info(f'Saved checkpoint with {len(results_dict)} entries')


def load_checkpoint(checkpoint_file='checkpoint.json'):
    """Load progress from checkpoint file"""
    if not os.path.exists(checkpoint_file):
        return {}, []

    with open(checkpoint_file) as f:
        data = json.load(f)

    logger.info(f'Loaded checkpoint from {data.get("timestamp")} with {len(data.get("summaries", {}))} entries')
    return data.get('summaries', {}), data.get('detailed_results', [])


def run_eval(
    df: pd.DataFrame,
    max_workers=4,
    checkpoint_interval=10,
    checkpoint_file='checkpoint.json',
) -> pd.DataFrame:
    # Load existing progress if available
    completed_results, detailed_results = load_checkpoint(checkpoint_file)

    # Filter out already processed rows
    processed_ids = set(str(v['id']) for v in completed_results.values())
    pending_rows = [(i, row) for i, row in df.iterrows() if str(row.get('id', i)) not in processed_ids]

    logger.info(
        f'Processing {len(pending_rows)} rows with {max_workers} workers ({len(processed_ids)} already completed)',
    )

    checkpoint_counter = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, row_data): row_data[0] for row_data in pending_rows}

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(pending_rows)):
            idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    row_id = str(result['summary']['id'])
                    completed_results[row_id] = result['summary']
                    detailed_results.append(result['detailed'])

                    # Save checkpoint periodically
                    checkpoint_counter += 1
                    if checkpoint_counter >= checkpoint_interval:
                        save_checkpoint(completed_results, detailed_results, checkpoint_file)
                        checkpoint_counter = 0
            except Exception as e:
                logger.error(f'Error processing row {idx}: {e!s}')

    # Final checkpoint
    if checkpoint_counter > 0:
        save_checkpoint(completed_results, detailed_results, checkpoint_file)

    # Save detailed results to file
    with open('../../../output/gsm8k_detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)

    return pd.DataFrame(list(completed_results.values()))


def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    df['pass1'] = df['pass1'].astype(int)
    out = []
    for col in ['logp_sum', 'avg_logp', 'perplexity']:
        pear, p1 = pearsonr(df[col], df['pass1'])
        spear, p2 = spearmanr(df[col], df['pass1'])
        out.append(dict(metric=col, pearson=pear, pearson_p=p1, spearman=spear, spearman_p=p2))
    return pd.DataFrame(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--workers', type=int, default=4, help='Number of concurrent workers')
    # parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save checkpoint every N items')
    # parser.add_argument('--checkpoint-file', type=str, default='gsm8k_checkpoint.json', help='Checkpoint file name')

    # args = parser.parse_args()

    base, easy, med, hard = create_gsm_evaluation_datasets(to_df=True)
    test_df = pd.concat(
        [
            base.assign(difficulty='base'),
            easy.assign(difficulty='easy'),
            med.assign(difficulty='medium'),
            hard.assign(difficulty='hard'),
        ],
    ).reset_index(drop=True)

    res = run_eval(
        test_df,
        max_workers=5,
        checkpoint_interval=10,
        checkpoint_file='gsm8k_checkpoint_test.json',
    )
    res.to_csv('gsm8k_pass1_logp.csv', index=False)

    corr = corr_table(res)
    corr.to_csv('metric_correlations.csv', index=False)
    logger.info('\n' + corr.to_string(index=False))
