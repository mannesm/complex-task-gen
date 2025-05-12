import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

model_path = "/gpfs/home6/mmokkenstorm/sync/qwen_models/finetuned_models/math_instruct/lora"

def load_model(model_path):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model, _ = FastLanguageModel.from_pretrained(model_path)

    # Set device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer, device

def generate_response(model, tokenizer, device, input_text, max_tokens: int= 1024):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


