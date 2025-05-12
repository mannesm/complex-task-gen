# train_gsm8k.py

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, default_data_collator
from unsloth import FastLanguageModel
from complex_task_gen.actual_project.constants import BASE_MATH_MODEL

# from complex_task_gen.actual_project.util.training_utils import CheckpointLoggerCallback

MODEL_NAME = BASE_MATH_MODEL  #TODO: Use instruct model
OUTPUT_DIR = f"./checkpoints/{MODEL_NAME}"

lora_config_default = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "target_modules": ["q_proj", "v_proj"]}


# === Load model + tokenizer ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
)

torch._dynamo.config.suppress_errors = True
# model = torch.compile(model)  # 🚀
torch.profiler.profile()

# === Add LoRA adapters ===
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config_default["r"],
    lora_alpha=lora_config_default["lora_alpha"],
    lora_dropout=lora_config_default["lora_dropout"],
    bias=lora_config_default["bias"],
    target_modules=lora_config_default["target_modules"],
)

# === Load and preprocess dataset ===
dataset = load_dataset("gsm8k", "main", split="train")

def format_example(example):
    question = example["question"].strip()
    answer = example["answer"].strip()
    sliced_answer = example["answer"].split("####")[-1].strip()
    return {
        "text": f"{question}"
                f"\n{answer}"
                f"\nAnswer: {sliced_answer}",
    }
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# === Tokenize dataset ===
def tokenize(example):
    # Tokenize everything as input_ids
    output = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    input_ids = output["input_ids"]

    # Create labels — mask prompt part (everything before answer)
    # Todo: Take an Instruct tuned model
    # TODO: System prompt --> the same
    # TODO: Question: Question
    # TODO: Answer: Answer --> full answer


    try:
        answer_start = tokenizer(example["text"]).input_ids.index(tokenizer.encode("\nAnswer:")[0]) + 1
        # TODO: Use ApplyChatTemplate
        # TODO: Return assistant_token_mask
    except ValueError:
        answer_start = 0  # fallback

    labels = input_ids.copy()
    labels[:answer_start] = [-100] * answer_start

    output["labels"] = labels
    return output

tokenized_dataset = dataset.map(tokenize)

# === Define training args ===
training_args = TrainingArguments(
    output_dir=f"./checkpoints/{MODEL_NAME}",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_steps=50,
    save_total_limit=20,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    bf16=True,
    report_to="none",
    warmup_steps=100,
    push_to_hub=False,
)

# === Trainer ===
trainer = Trainer( #TODO: Check if the input data is correct --> take the tokenized dataset and check if the tokenized output looks fine
    model=model, # TODO: Check if the ChatTemplate is correct
    args=training_args,  #TODO: Add some kind of accuray
    train_dataset=tokenized_dataset, #TODO: add logging on Weight and bias?
    data_collator=default_data_collator
)


# === Train ===
trainer.train()

# === Save final model ===
trainer.save_model(f"./checkpoints/{MODEL_NAME}/final")
tokenizer.save_pretrained(f"./checkpoints/{MODEL_NAME}/final")
model.config.save_pretrained(f"./checkpoints/{MODEL_NAME}/final")
print("✅ Fine-tuning complete. Model saved.")


from huggingface_hub import upload_folder
import os

os.chdir("/home/mmokkenstorm/sync/")


upload_folder(
    folder_path="./checkpoints/gsm8k-qwen1.5b/final",
    repo_id="Mannesmok/qwen_finetune_base_gsm8k",
    repo_type="model"
)

