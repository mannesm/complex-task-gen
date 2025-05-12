from datasets import load_dataset
from openai import OpenAI


model_name = "Qwen/Qwen2.5-Math-7B"


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)


gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="train")

gsm_symbolic_easy = load_dataset("apple/GSM-Symbolic", name="main")

gsm_symbolic_medium = load_dataset("apple/GSM-Symbolic", name="p1")

gsm_symbolic_hard = load_dataset("apple/GSM-Symbolic", name="p2")

math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="train")
