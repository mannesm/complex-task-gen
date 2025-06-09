#!/usr/bin/env python
# train_qwen_gsm8k.py
# ------------------------------------------------------------
# Fine tunes a Qwen-Chat model for chain-of-thought math reasoning
# (GSM8K format) using LoRA + PEFT via Unsloth.
#
# Tested with:
#   unsloth==0.6.2
#   transformers>=4.40
#   datasets>=2.19
#   bitsandbytes>=0.43
# ------------------------------------------------------------
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd

# Then import torch or tensorflow, etc.
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from unsloth import FastLanguageModel

BASE_MATH_MODEL = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
# ---------------------------------------------------------------------
# 0. CLI
# ---------------------------------------------------------------------

training_epochs = 3
batch_size = 64
accum_steps = 8  # "Gradient accumulation to reach effective batch."
learning_rate = 2e-4
max_len = 1024

output_dir = '/home/mmokkenstorm/sync/qwen_models/finetuned_models/n30_best/math_instruct_chattemplatetrue'
MODEL_NAME = BASE_MATH_MODEL


# ---------------------------------------------------------------------
# 1. System prompt
# ---------------------------------------------------------------------
SYSTEM_PROMPT = (
    'You are a helpful assistant. Solve the math problem step by step. '
    'The answer should be a number, and you should always return it in the format: '
    'The final answer is: <answer> 42 </answer>.'
).strip()

print('🔹 Loading base model & tokenizer …')
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_len,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map='auto',
    quantization_config=bnb_cfg,
)

# Add LoRA adapters
print('🔹 Injecting LoRA adapters …')
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj'],
)

# ---------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------
print('🔹 Loading GSM8K …')
# raw = load_dataset('gsm8k', 'main')
# Create a sample DataFrame
# df['question'] = df['task']
df = pd.read_csv(
    '/gpfs/home6/mmokkenstorm/augmentation_outputs/N_SAMPLES_30_N_AUGS_1/2025-06-08 18:43:21/checkpoint_best_7460.csv',
)
df['answer'] = df['solution']
df['question'] = df['task']
# Convert to Hugging Face Dataset
# Filter df where passed = True
# df = df[df['passed'] == True]
hf_dataset = Dataset.from_pandas(df)
hf_dataset
# Inspect the dataset
print(hf_dataset)
raw = DatasetDict({'train': hf_dataset})
raw
raw = raw.remove_columns([c for c in raw['train'].column_names if c not in ('question', 'answer')])
raw


def build_chat(example):
    """Convert GSM8K row → ChatML conversation string."""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': example['question'].strip()},
        {'role': 'assistant', 'content': example['answer'].strip()},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {'text': chat}


raw['train']
raw = raw.map(build_chat, desc='💬 Building chat examples')

# ── split train/val (5 % dev) ────────────────────────────────────────
train_val = raw['train'].train_test_split(test_size=0.05, seed=42, shuffle=True)
data = DatasetDict(
    {
        'train': train_val['train'],
        'validation': train_val['test'],
    },
)

data


# ---------------------------------------------------------------------
# 4. Tokenisation & label masking
# ---------------------------------------------------------------------
def mask_labels(example):
    encoded = tokenizer(
        example['text'],
        truncation=True,
        padding='max_length',
        max_length=max_len,
    )
    input_ids = encoded['input_ids']

    # Mask tokens up to (and incl.) first <|im_start|>assistant tag
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    try:
        first_im_start = input_ids.index(im_start_id)  # system tag
        second_im_start = input_ids.index(im_start_id, first_im_start + 1)
        assistant_start = input_ids.index(im_start_id, second_im_start + 1)
    except ValueError:
        assistant_start = 0  # should not happen

    labels = [-100] * (assistant_start + 1) + input_ids[assistant_start + 1 :]
    encoded['labels'] = labels[:max_len]
    return encoded


raw

tokenised = data.map(
    mask_labels,
    remove_columns=['text'],
    desc='🔑 Tokenising',
)

# ---------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------
print('🔹 Preparing trainer …')
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=accum_steps,
    num_train_epochs=training_epochs,
    learning_rate=learning_rate,
    lr_scheduler_type='cosine',
    warmup_steps=100,
    bf16=True,
    logging_steps=10,
    save_steps=250,
    eval_steps=250,
    eval_strategy='steps',
    save_total_limit=3,
    optim='paged_adamw_32bit',
    report_to='none',
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised['train'],
    eval_dataset=tokenised['validation'],
    data_collator=default_data_collator,
)

print('🚀 Starting fine-tuning …')
trainer.train()

# ---------------------------------------------------------------------
# 6. Save artefacts
# ---------------------------------------------------------------------
print('💾 Saving LoRA adapter and tokenizer …')
adapter_path = Path(output_dir) / 'lora'
adapter_path.mkdir(parents=True, exist_ok=True)
# Unsloth provides a helper for adapter-only save:
model.save_pretrained(adapter_path, safe_serialization=True)

tokenizer.save_pretrained(adapter_path)

print('✅ Done.  Adapter written to', adapter_path.resolve())

# model, _ = FastLanguageModel.from_pretrained(model_path)  # Unpack if it returns a tuple
# Load the model and tokenizer

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

model_path = '/gpfs/home6/mmokkenstorm/sync/qwen_models/finetuned_models/math_instruct/lora'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model, _ = FastLanguageModel.from_pretrained(model_path)

# Set device and move model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Prepare inputs and move to same device
input_text = 'Solve: What is 12 + 8?'
inputs = tokenizer(input_text, return_tensors='pt').to(device)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=500)

# Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
