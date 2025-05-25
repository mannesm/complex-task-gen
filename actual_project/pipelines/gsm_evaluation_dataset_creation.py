import random
import re

from datasets import load_dataset


def create_gsm_evaluation_datasets(to_df=False):
    random.seed(42)

    symbolic_easy = load_dataset('apple/GSM-Symbolic', 'main')
    symbolic_medium = load_dataset('apple/GSM-Symbolic', 'p1')
    symbolic_hard = load_dataset('apple/GSM-Symbolic', 'p2')
    gsm8k = load_dataset('openai/gsm8k', 'main', split='test')
    gsm8k = gsm8k.map(lambda x, idx: {**x, 'original_id': int(idx)}, with_indices=True)

    original_q_list_hard = list(set(symbolic_hard['test']['original_id']))
    original_q_list_medium = list(set(symbolic_medium['test']['original_id']))
    original_q_list_easy = list(set(symbolic_easy['test']['original_id']))
    common_ids = set(original_q_list_hard) & set(original_q_list_medium) & set(original_q_list_easy)

    def select_random_entry(filtered_data):
        grouped = {}
        for item in filtered_data:
            grouped.setdefault(item['original_id'], []).append(item)
        return [random.choice(entries) for entries in grouped.values()]

    def extract_solution(answer: str) -> str:
        match = re.search(r'####\s*([0-9\.,\-]+)', answer)
        return match.group(1).replace(',', '') if match else None

    filtered_easy = [item for item in symbolic_easy['test'] if item['original_id'] in common_ids]
    filtered_medium = [item for item in symbolic_medium['test'] if item['original_id'] in common_ids]
    filtered_hard = [item for item in symbolic_hard['test'] if item['original_id'] in common_ids]
    selected_gsm8k = [gsm8k[int(idx)] for idx in common_ids]

    selected_easy = select_random_entry(filtered_easy)
    selected_medium = select_random_entry(filtered_medium)
    selected_hard = select_random_entry(filtered_hard)

    def process_items(items):
        items = sorted(items, key=lambda x: x['original_id'])
        return [
            {
                'original_id': item['original_id'],
                'question': item['question'],
                'answer': item['answer'],
                'solution': extract_solution(item['answer']),
            }
            for item in items
        ]

    selected_easy = process_items(selected_easy)
    selected_medium = process_items(selected_medium)
    selected_hard = process_items(selected_hard)

    selected_gsm8k = sorted(selected_gsm8k, key=lambda x: int(x['original_id']))
    selected_gsm8k = [
        {
            'original_id': item['original_id'],
            'question': item['question'],
            'answer': item['answer'],
            'solution': extract_solution(item['answer']),
        }
        for item in selected_gsm8k
    ]

    if to_df:
        import pandas as pd

        selected_easy = pd.DataFrame(selected_easy)
        selected_medium = pd.DataFrame(selected_medium)
        selected_hard = pd.DataFrame(selected_hard)
        selected_gsm8k = pd.DataFrame(selected_gsm8k)

    return selected_gsm8k, selected_easy, selected_medium, selected_hard


def create_full_gsm8k_test_dataset(to_df=False):
    # Load GSM8K test set
    gsm8k = load_dataset('openai/gsm8k', 'main', split='test')
    gsm8k = gsm8k.map(lambda x, idx: {**x, 'original_id': int(idx)}, with_indices=True)

    def extract_solution(answer: str) -> str:
        match = re.search(r'####\s*([0-9\.,\-]+)', answer)
        return match.group(1).replace(',', '') if match else None

    processed_gsm8k = [
        {
            'original_id': item['original_id'],
            'question': item['question'],
            'answer': item['answer'],
            'solution': extract_solution(item['answer']),
        }
        for item in sorted(gsm8k, key=lambda x: int(x['original_id']))
    ]

    if to_df:
        import pandas as pd

        processed_gsm8k = pd.DataFrame(processed_gsm8k)

    return processed_gsm8k
