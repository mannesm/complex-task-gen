from transformers import TrainingArguments

OUTPUT_DIR = "./checkpoints/gsm8k-qwen1.5b"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps= 40,
    save_steps=5,
    save_total_limit=20,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    bf16=True,
    report_to="none",
    warmup_steps=100,
    push_to_hub=False,
)


lora_config_default = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "target_modules": ["q_proj", "v_proj"]}
