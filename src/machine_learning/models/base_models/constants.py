MODEL_NAMES = {
    "QWEN_2_MATH_7B": "qwen2_7b_math_fp",
    "LLAMA_3_2": "llama_3_2",
    "DEEPSEEK_R1": "deepseek_r1",
}

MODEL_CONFIG = {
    "DEFAULT": {"temperature": 0.5, "max_tokens": 256},
    "QWEN_2_MATH_7B": {"temperature": 0.5, "max_tokens": 256},
    "LLAMA_3_2": {},
    "DEEPSEEK_R1": {},
}

TOKEN_MAX_DEFAULT = 1024

HUGGINGFACE_MODEL_NAMES = {
    "QWEN_2_MATH_7B": "Qwen/Qwen2.5-Math-7B-Instruct",
}