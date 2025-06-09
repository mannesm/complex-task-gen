#!/usr/bin/env python
# Updated Qwen2.5 Math Fine-Tuning Script using Unsloth + JSONL + ChatML

import json
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from unsloth import FastLanguageModel

# ---------------------------- Configuration ----------------------------
CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

BASE_MATH_MODEL = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
OUTPUT_DIR = '/home/mmokkenstorm/sync/qwen_models/finetuned_models/n30_best/math_instruct_chattemplatetrue_v2'
DATA_CSV = (
    '/gpfs/home6/mmokkenstorm/augmentation_outputs/N_SAMPLES_30_N_AUGS_1/2025-06-08 18:43:21/checkpoint_best_7460.csv'
)
JSONL_PATH = '/gpfs/home6/mmokkenstorm/data/qwen2.5_math_finetune_chatml.jsonl'

# ------------------------- Generate JSONL ------------------------------
df = pd.read_csv(DATA_CSV)

system_prompt = (
    'You are a helpful assistant. Solve the math problem step by step. '
    'The answer should be a number, and you should always return it in the format: '
    'The final answer is: <answer> 42 </answer>.'
)


def format_example(row):
    try:
        answer_line = row['solution'].strip().splitlines()[-1]
        last_number = [float(tok) for tok in answer_line.replace(',', '').split() if tok.replace('.', '', 1).isdigit()]
        final = last_number[-1] if last_number else ''
    except Exception:
        final = ''

    return {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': row['task'].strip()},
            {
                'role': 'assistant',
                'content': row['solution'].strip() + f'\n\nThe final answer is: <answer> {final} </answer>',
            },
        ],
    }


formatted = [format_example(row) for _, row in df.iterrows()]

with open(JSONL_PATH, 'w') as f:
    for ex in formatted:
        f.write(json.dumps(ex) + '\n')

# ---------------------------- Load Model -------------------------------
print('🔹 Loading base model & tokenizer …')
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MATH_MODEL,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map='auto',
    quantization_config=bnb_cfg,
)

# ---------------------------- Add LoRA ---------------------------------
print('🔹 Injecting LoRA adapters …')
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj'],
)

# --------------------------- Load Dataset ------------------------------
print('🔹 Loading formatted JSONL dataset …')
raw = load_dataset('json', data_files={'train': JSONL_PATH})
raw = raw['train'].train_test_split(test_size=0.05, seed=42)
data = DatasetDict({'train': raw['train'], 'validation': raw['test']})


def apply_chat_template(example):
    chat = tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {'text': chat}


data = data.map(apply_chat_template, desc='💬 Applying chat templates')
data


# ---------------------- Tokenization & Masking -------------------------
def mask_labels(example):
    encoded = tokenizer(
        example['text'],
        truncation=True,
        padding='max_length',
        max_length=1024,
    )
    input_ids = encoded['input_ids']
    # Mask all tokens before assistant response
    try:
        assistant_index = input_ids.index(tokenizer.convert_tokens_to_ids('<|im_start|>'))
        assistant_index = input_ids.index(
            tokenizer.convert_tokens_to_ids('<|im_start|>'),
            assistant_index + 1,
        )
        assistant_index = input_ids.index(
            tokenizer.convert_tokens_to_ids('<|im_start|>'),
            assistant_index + 1,
        )
    except ValueError:
        assistant_index = 0

    encoded['labels'] = [-100] * (assistant_index + 1) + input_ids[assistant_index + 1 :]
    encoded['labels'] = encoded['labels'][:1024]
    return encoded


data = data.map(mask_labels, remove_columns=['text', 'messages'], desc='🔑 Tokenising')

data
# --------------------------- Training ----------------------------------
print('🔹 Preparing trainer …')
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
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
    train_dataset=data['train'],
    eval_dataset=data['validation'],
    data_collator=default_data_collator,
)

print('🚀 Starting fine-tuning …')
trainer.train()

# ----------------------------- Save ------------------------------------
print('💾 Saving LoRA adapter and tokenizer …')
adapter_path = Path(OUTPUT_DIR) / 'lora'
adapter_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(adapter_path, safe_serialization=True)
tokenizer.save_pretrained(adapter_path)
print('✅ Done.  Adapter written to', adapter_path.resolve())
