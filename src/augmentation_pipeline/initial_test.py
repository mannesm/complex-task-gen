from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch


QWEN_MATH_MODEL = "Qwen/Qwen2.5-Math-7B"

REWARD_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"





gen_tok = AutoTokenizer.from_pretrained(QWEN_MATH_MODEL, trust_remote_code=True)
gen_lm  = AutoModelForCausalLM.from_pretrained(QWEN_MATH_MODEL,
                                               torch_dtype=torch.float16).cuda()

REWARD_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"

rew_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B", trust_remote_code=True)
rew_model = AutoModel.from_pretrained(              # <─ NOTE AutoModel !
          REWARD_MODEL,
          device_map="cuda",                  # fits in ~6 GB fp16
          torch_dtype=torch.bfloat16,         # or fp16
          trust_remote_code=True).eval()


def score(prompt, completion):
    """Return a scalar reward."""
    text = prompt + completion
    ids  = rew_tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        return rew_mdl(**ids).logits.item()          # ↑ good

def generate_one(seed):
    g_ids = gen_tok(seed, return_tensors="pt").to("cuda")
    out   = gen_lm.generate(**g_ids,
                            max_new_tokens=256,
                            temperature=0.9,
                            top_p=0.95,
                            num_return_sequences=4)
    cands = [gen_tok.decode(o[g_ids["input_ids"].shape[-1]:], skip_special_tokens=True)
             for o in out]
    # Re-rank by reward
    ranked = sorted(cands, key=lambda c: score(seed, c), reverse=True)
    return ranked[0]      # or keep top-k above τ
