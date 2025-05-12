import json
import gzip
import pathlib
import random
import time
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
import logging
logging.basicConfig(level=logging.INFO)

# ─────────────────────────  CONFIG  ──────────────────────────────────
GEN_MODEL      = "Qwen/Qwen2.5-Math-7B"
REWARD_MODEL   = "Qwen/Qwen2.5-Math-PRM-7B"
N_SEEDS        = 100
N_CAND         = 4
OUT_PATH       = pathlib.Path("aug_gsm8k.jsonl.gz")
SEED           = 42
LOG_EVERY      = 5             # print a progress line every N seeds
# ─────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
random.seed(SEED)
torch.manual_seed(SEED)

logging.info("[info] loading generator …")
gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL, trust_remote_code=True)
gen_lm  = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL, torch_dtype=torch.float16, device_map="auto"
).eval()

logging.info("[info] loading reward model …")
rew_tok = AutoTokenizer.from_pretrained(REWARD_MODEL, trust_remote_code=True)
rew_mdl = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL, torch_dtype=torch.float16, device_map="auto"
).eval()

@torch.no_grad()
def reward_score(prompt, compl):
    ids = rew_tok(prompt + compl, return_tensors="pt").to(DEVICE)
    return rew_mdl(**ids).logits[0, 1].item()

# 2️⃣  seed data
gsm = load_dataset("gsm8k", "main", split="train")
seed_ids = random.sample(range(len(gsm)), k=N_SEEDS)

# 3️⃣  writer
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fout = gzip.open(OUT_PATH, "wt", encoding="utf-8")
logging.info(f"[info] writing JSON-Lines to {OUT_PATH.resolve()}\n")

t0 = time.time()
for n_done, idx in enumerate(seed_ids, 1):
    seed = gsm[idx]
    q, a = seed["question"], seed["answer"]
    logging.info(f"[seed {n_done}/{N_SEEDS}] Q0 → {q[:60]}…")

    prompt = (
        "Below is an example of a maths word problem solved step-by-step.\n\n"
        f"Example\nQ: {q}\nA: {a}\n\n"
        "Now create a **new** problem that is slightly more difficult but of the same type. "
        "Provide full reasoning and end with '#### <final answer>'.\n\nQ:"
    )

    in_ids = gen_tok(prompt, return_tensors="pt").to(DEVICE)
    outs = gen_lm.generate(
        **in_ids,
        max_new_tokens=512,
        temperature=0.9, top_p=0.95,
        num_return_sequences=N_CAND, do_sample=True,
        pad_token_id=gen_tok.eos_token_id,
    )
    cands = [
        gen_tok.decode(o[in_ids["input_ids"].shape[-1]:], skip_special_tokens=True)
        for o in outs
    ]

    scored = [{"text": c, "reward": reward_score(prompt, c)} for c in cands]
    best = max(scored, key=lambda d: d["reward"])

    # console-log a terse summary
    logging.info(f"  ↳ generated {len(scored)} cands; best reward = {best['reward']:.3f}")

    # save
    record = {
        "seed_id": int(idx),
        "timestamp": datetime.now().isoformat(timespec="seconds") + "Z",
        "prompt": prompt,
        "candidates": scored,
        "best": best,
    }
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # periodic progress line
    if n_done % LOG_EVERY == 0 or n_done == N_SEEDS:
        elapsed = time.time() - t0
        logging.info(f"[progress] {n_done}/{N_SEEDS} seeds done — {elapsed/60:.1f} min elapsed\n")

fout.close()
logging.info("[✓] finished — file saved, you can now load it with datasets.load_dataset('json', …)")



