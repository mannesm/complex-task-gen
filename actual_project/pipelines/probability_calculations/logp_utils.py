# logp_utils.py
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
