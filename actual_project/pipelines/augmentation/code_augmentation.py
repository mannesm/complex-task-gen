import numpy as np
import requests
from openai import OpenAI

from pipelines.augmentation.augmentation_prompts import (
    text2code_examples,
    text2code_prompt,
    augmentation_examples,
    augmentation_prompt,
)


config = {
    'text2code_examples': text2code_examples,
    'text2code_prompt': text2code_prompt,
    'augmentation_examples': augmentation_examples,
    'augmentation_prompt': augmentation_prompt,
    'n_iterations': 10,
}


model_name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')

prompt = 'Mary is an avid gardener. Yesterday, she received 18 new potted plants from her favorite plant nursery. She already has 2 potted plants on each of the 40 window ledges of her large country home. Feeling generous, she has decided that she will give 1 potted plant from each ledge to friends and family tomorrow. How many potted plants will Mary remain with?'
response = requests.post(
    'http://localhost:8000/v1/completions',
    json={
        'model': model_name,
        'prompt': prompt,
        'max_tokens': 1024,
        'temperature': 0.0,
        'logprobs': 1,
    },
)
response_data = response.json()
log_probs = response_data['choices'][0]['logprobs']['token_logprobs']
output_text = response_data['choices'][0]['text']
sum_log_p = sum(log_probs)

num_tokens = len(log_probs)
avg_log_p = sum_log_p / num_tokens
perplexity = np.exp(-avg_log_p)

print(f"Novelty score for the prompt '{prompt}': {perplexity}")
