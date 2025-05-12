import requests
from openai import OpenAI

from pipelines.augmentation.augmentation_prompts import (
    text2code_examples,
    text2code_prompt,
    augmentation_examples,
    augmentation_prompt,
)

# TODO: Implement code augmentation Pipeline

config = {
    'text2code_examples': text2code_examples,
    'text2code_prompt': text2code_prompt,
    'augmentation_examples': augmentation_examples,
    'augmentation_prompt': augmentation_prompt,
    'n_iterations': 10,
}


model_name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')

prompt = 'Hello, how are you?'
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
novelty_score = sum(log_probs)  # Aggregate log probabilities (e.g., sum or mean)


print(f"Novelty score for the prompt '{prompt}': {novelty_score}")
