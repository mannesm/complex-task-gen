import os
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from transformers import Trainer, TrainingArguments, default_data_collator
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ---------------------------- Configuration ----------------------------
OUTPUT_DIR = '/home/mmokkenstorm/sync/qwen_models/finetuned_models/n_gsm8k/math_instruct_chattemplatetrue_v2'
EPOCHS = 3
LEARNING_RATE = 1e-4  # for full FT; use 3e-5 for LoRA
BATCH_SIZE = 32
GRAD_ACCUM = 8
MAX_LENGTH = 1024
FULL_FT = True  # set False for 16-bit LoRA

BASE_MODEL = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
SYSTEM_PROMPT = (
    'You are a helpful assistant. Solve the math problem step by step. '
    'The answer should be a number, and you should always return it in the format: '
    'The final answer is: <answer> 42 </answer>.'
)

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def build_messages_gsm8k(split='train'):
    ds = load_dataset('openai/gsm8k', 'main', split=split)
    msgs = []
    for row in ds:
        try:
            ans_line = row['answer'].strip().splitlines()[-1]
            nums = [tok for tok in ans_line.replace(',', '').split() if tok.replace('.', '', 1).isdigit()]
            final = nums[-1] if nums else ''
        except Exception:
            final = ''
        msgs.append(
            {
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': row['question'].strip()},
                    {
                        'role': 'assistant',
                        'content': f'{row["answer"].strip()}\n\nThe final answer is: <answer> {final} </answer>',
                    },
                ],
            },
        )
    return msgs


print('Reading GSM8K and building message list …')
train_msgs = build_messages_gsm8k('train')
val_msgs = build_messages_gsm8k('test')

print('Creating HF Datasets …')
from datasets import Dataset

datasets = DatasetDict(
    {
        'train': Dataset.from_list(train_msgs),
        'validation': Dataset.from_list(val_msgs),
    },
)

print('Loading tokenizer & applying Qwen ChatML template …')
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

print('Starting training …')
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
