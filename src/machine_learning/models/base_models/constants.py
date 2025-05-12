MODEL_NAMES = {
    "QWEN_2_MATH_7B": "qwen2_7b_math_fp",
    "QWEN_2_15B": "qwen2.5:1.5b",
    "LLAMA_3_2": "llama_3_2",
    "DEEPSEEK_R1": "deepseek_r1",
}

MODEL_CONFIG = {
    "DEFAULT": {"temperature": 0.8, "max_tokens": 256},
    "QWEN_2_MATH_7B": {"temperature": 0.5, "max_tokens": 256},
    "QWEN_2_15B": {"temperature": 0.8, "max_tokens": 256},
    "LLAMA_3_2": {},
    "DEEPSEEK_R1": {},
}

TOKEN_MAX_EFAULT = 1024

HUGGINGFACE_MODEL_NAMES = {
    "QWEN_2_MATH_7B": "Qwen/Qwen2.5-Math-7B-Instruct",
    "QWEN_2_15B": "Qwen/Qwen2.5-1.5B",
    "QWEN_25_CODER_7B": "Qwen/Qwen2.5-Coder-7B-Instruct"
}

OLLAMA = "ollama"
VLMM = "vllm"
