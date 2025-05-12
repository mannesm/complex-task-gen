import torch
import json
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, default_data_collator
from unsloth import FastLanguageModel
from pathlib import Path
import matplotlib.pyplot as plt

from complex_task_gen.actual_project.constants import MODEL_OUTPUT_DIR

# === Load tokenizer and dataset ===
base_dir = Path(MODEL_OUTPUT_DIR)  # <-- UPDATE THIS

model_name = base_dir / "final"  # use tokenizer from final model
tokenizer = FastLanguageModel.from_pretrained(model_name)[1]

dataset = load_dataset("gsm8k", "main", split="test")

def format_example(example):
    question = example["question"].strip()
    answer = example["answer"].split("####")[-1].strip()
    return {"text": f"Question: {question}\nAnswer: {answer}"}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

def tokenize(example):
    output = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    input_ids = output["input_ids"]

    try:
        answer_start = tokenizer(example["text"]).input_ids.index(tokenizer.encode("\nAnswer:")[0]) + 1
    except ValueError:
        answer_start = 0

    labels = input_ids.copy()
    labels[:answer_start] = [-100] * answer_start
    output["labels"] = labels
    return output

tokenized_dataset = dataset.map(tokenize)

# === Load checkpoint metadata ===
with open(base_dir / "checkpoints.json") as f:
    checkpoints = json.load(f)

results = []

for ckpt in checkpoints:
    ckpt_path = base_dir / ckpt["checkpoint_dir"]
    print(f"📍 Evaluating {ckpt_path}")

    model, _ = FastLanguageModel.from_pretrained(
        model_name=str(ckpt_path),
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=64,
            bf16=True,
            dataloader_drop_last=False,
        ),
        eval_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )

    metrics = trainer.evaluate()
    results.append({"step": ckpt["global_step"], "eval_loss": metrics["eval_loss"]})

# === Plot learning curve ===
steps = [r["step"] for r in results]
losses = [r["eval_loss"] for r in results]

plt.plot(steps, losses, marker="o")
plt.xlabel("Training Step")
plt.ylabel("Validation Loss")
plt.title("Learning Curve on GSM8K")
plt.grid(True)
plt.show()
