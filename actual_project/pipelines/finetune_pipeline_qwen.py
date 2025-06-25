import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ---------------------------- Configuration ----------------------------
OUTPUT_DIR = '/home/mmokkenstorm/sync/qwen_models/finetuned_models/n30_best/math_instruct_chattemplatetrue_v2'
CSV_PATH = (
    '/gpfs/home6/mmokkenstorm/augmentation_outputs/N_SAMPLES_30_N_AUGS_1/2025-06-08 18:43:21/checkpoint_best_7460.csv'
)
EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GRAD_ACCUM = 8
MAX_LENGTH = 1024
FULL_FT = True

BASE_MODEL = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
SYSTEM_PROMPT = (
    'You are a helpful assistant. Solve the math problem step by step. '
    'The answer should be a number, and you should always return it in the format: '
    'The final answer is: <answer> 42 </answer>.'
)

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def build_messages(csv_path: str):
    df = pd.read_csv(csv_path)
    msgs = []
    for _, row in df.iterrows():
        try:
            ans_line = row['solution'].strip().splitlines()[-1]
            nums = [tok for tok in ans_line.replace(',', '').split() if tok.replace('.', '', 1).isdigit()]
            final = nums[-1] if nums else ''
        except Exception:
            final = ''
        msgs.append(
            {
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': row['task'].strip()},
                    {
                        'role': 'assistant',
                        'content': f'{row["solution"].strip()}\n\nThe final answer is: <answer> {final} </answer>',
                    },
                ],
            },
        )
    return msgs


print(' Reading CSV and building message list …')
all_msgs = build_messages(CSV_PATH)

print('Creating HF Datasets …')
raw_ds = Dataset.from_list(all_msgs)
raw_split = raw_ds.train_test_split(test_size=0.05, seed=42)
datasets = DatasetDict({'train': raw_split['train'], 'validation': raw_split['test']})

print('🔡 Loading tokenizer & applying Qwen ChatML template …')
model, tok = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    device_map='auto',
)
tok = get_chat_template(tok, chat_template='qwen-2.5')


def apply_and_tokenize(example):
    wrapped = tok.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tok(
        wrapped,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )
    tag = '<|im_start|>assistant'
    cut_tokens = len(tok(wrapped.split(tag)[0], add_special_tokens=False)['input_ids']) + 1
    enc['labels'] = [-100] * cut_tokens + enc['input_ids'][cut_tokens:]
    return enc


print('🧹 Tokenising & masking …')
processed = datasets.map(
    apply_and_tokenize,
    remove_columns=['messages'],
    desc='Tokenising',
)

print('🛠  Configuring fine-tuning method …')
if not FULL_FT:
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj'],
    )
print('🚀 Starting training …')
train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    bf16=True,
    optim='adamw_torch',
    gradient_checkpointing=True,
    logging_steps=25,
    save_steps=250,
    eval_steps=250,
    # evaluation_strategy='steps',
    save_total_limit=3,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=processed['train'],
    eval_dataset=processed['validation'],
    data_collator=default_data_collator,
)

trainer.train()

print('Saving model & tokenizer …')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tok.save_pretrained(OUTPUT_DIR)
print('Done — finetuned model written to', Path(OUTPUT_DIR).resolve())
