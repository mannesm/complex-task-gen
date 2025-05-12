from datasets import load_dataset
import random


def create_gsm_evaluation_datasets(to_df=False):
    random.seed(42)

    # Load all datasets
    symbolic_easy = load_dataset("apple/GSM-Symbolic", "main")
    symbolic_medium = load_dataset("apple/GSM-Symbolic", "p1")
    symbolic_hard = load_dataset("apple/GSM-Symbolic", "p2")
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k = gsm8k.map(lambda x, idx: {**x, "original_id": int(idx)}, with_indices=True)


    original_q_list_hard = list(set(symbolic_hard['test']['original_id']))
    original_q_list_medium = list(set(symbolic_medium['test']['original_id']))
    original_q_list_easy = list(set(symbolic_easy['test']['original_id']))
    gsm8k_ids = list(set(gsm8k['original_id']))
    common_ids = set(original_q_list_hard) & set(original_q_list_medium) & set(original_q_list_easy)

    # Filter datasets based on common_ids
    filtered_easy = [item for item in symbolic_easy['test'] if item['original_id'] in common_ids]
    filtered_medium = [item for item in symbolic_medium['test'] if item['original_id'] in common_ids]
    filtered_hard = [item for item in symbolic_hard['test'] if item['original_id'] in common_ids]

    # Group by original_id and randomly select one entry per group
    def select_random_entry(filtered_data):
        grouped = {}
        for item in filtered_data:
            grouped.setdefault(item['original_id'], []).append(item)
        return [random.choice(entries) for entries in grouped.values()]

    selected_gsm8k = [gsm8k[int(idx)] for idx in common_ids]
    selected_easy = select_random_entry(filtered_easy)
    selected_medium = select_random_entry(filtered_medium)
    selected_hard = select_random_entry(filtered_hard)

    # Sort each dataset by original_id
    selected_easy = sorted(selected_easy, key=lambda x: x['original_id'])
    selected_easy = [
        {"original_id": item["original_id"], "question": item["question"], "answer": item["answer"]}
        for item in selected_easy
    ]

    selected_medium = sorted(selected_medium, key=lambda x: x['original_id'])
    selected_medium = [
        {"original_id": item["original_id"], "question": item["question"], "answer": item["answer"]}
        for item in selected_medium
    ]

    selected_hard = sorted(selected_hard, key=lambda x: x['original_id'])
    selected_hard = [
        {"original_id": item["original_id"], "question": item["question"], "answer": item["answer"]}
        for item in selected_hard
    ]

    selected_gsm8k = sorted(selected_gsm8k, key=lambda x: int(x['original_id']))
    selected_gsm8k = [
        {"original_id": item["original_id"], "question": item["question"], "answer": item["answer"]}
        for item in selected_gsm8k
    ]
    # print(selected_easy[0])
    # print(selected_medium[0])
    # print(selected_hard[0])
    # print(selected_gsm8k[0])

    if to_df:
        import pandas as pd
        selected_easy = pd.DataFrame(selected_easy)
        selected_medium = pd.DataFrame(selected_medium)
        selected_hard = pd.DataFrame(selected_hard)
        selected_gsm8k = pd.DataFrame(selected_gsm8k)
    return selected_gsm8k, selected_easy, selected_medium, selected_hard


