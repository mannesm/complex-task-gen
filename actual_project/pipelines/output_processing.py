import logging
import re
from typing import Dict

import pandas as pd
from constants import DATASET_LIST

dataframes = {}
output_prediction_patterns = [
    r'\\\(\s*\\boxed{(\d+)}\s*\\\)',
    r'\[[\s\S]*?\\boxed{(\d+)}[\s\S]*?\]',
    r'\\boxed{(\d+)}',
    r'{{\d}}',
]


def load_model_output(
    dataset_list: list = DATASET_LIST,
    folder_path: str = '/Users/mannes/PycharmProjects/thesis_project_2/model_outputs/base_finetune',
) -> Dict[str, pd.DataFrame]:
    """
    Load datasets from CSV files.
    :param folder_path: Path to the CSV files Folder
    :param dataset_list: List of dataset names to load.
    :return: Dictionary of DataFrames.
    """
    dataframes = {}
    for name in dataset_list:
        path = f'{folder_path}/{name}.csv'
        try:
            df = pd.read_csv(path, sep='|')
            dataframes[name] = df  # Assign the DataFrame directly
            logging.info(f'Loaded {name} dataset')
        except FileNotFoundError:
            print(f'File not found: {path}')
    return dataframes


def extract_generated_answer(
    example: Dict[str, str], regex_patterns: list[str] = [r'####\s*(\d+)']
) -> str | None:
    """
    Extracts an answer from the example using a list of regex patterns.

    :param example: A dictionary containing the "answer" key.
    :param regex_patterns: A list of regex patterns to match against the answer.
    :return: The first matched group or None if no match is found.
    """
    answer = example['answer']
    for pattern in regex_patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(1)
    logging.info(
        f'Failed to match answer value for pattern: {regex_patterns} to {answer}'
    )
    return None


model_output_df_dict = load_model_output(DATASET_LIST)

for dataset_name, dataset in model_output_df_dict.items():
    print(f'Processing dataset: {dataset_name}')
    dataset['extracted_actual_answer'] = dataset.apply(extract_generated_answer, axis=1)
    dataset['extracted_generated_answer'] = dataset.apply(
        lambda x: extract_generated_answer(
            {'answer': x['generated_answer']}, output_prediction_patterns
        ),
        axis=1,
    )
    dataset['correct'] = dataset.apply(
        lambda x: x['extracted_actual_answer'] == x['extracted_generated_answer'],
        axis=1,
    )
    model_output_df_dict[dataset_name] = dataset
    print(f'Processed dataset: {dataset_name}')
    print(dataset['correct'].value_counts())
