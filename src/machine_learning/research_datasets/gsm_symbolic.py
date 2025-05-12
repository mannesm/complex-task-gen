from datasets import load_dataset
import pandas as pd
import re
import json

from machine_learning.research_datasets.base_dataset import BaseDataset


class GSMSymbolic(BaseDataset):
    def __init__(self, split="test"):
        self.dataset = load_dataset("apple/GSM-Symbolic", "main", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)

    def to_dataframe(self):
        return pd.DataFrame(self.dataset)


def download_gsm_symbolic_dataset(number_of_problems=5000):
    try:
        # Load the dataset using the datasets library
        dataset = load_dataset("apple/GSM-Symbolic", "main")

        # Get the items from train split
        data = dataset['test'].select(range(number_of_problems))

        # Convert to the desired format
        formatted_data = []
        for idx, item in enumerate(data):
            # Extract the final answer from the solution
            solution = item['answer']
            if solution:
                # GSM8K solutions typically end with "#### number"
                match = re.search(r'####\s*(\d+)', solution)
                if match:
                    number = match.group(1)
                    # Replace the "#### number" with "\boxed{number}"
                    solution = re.sub(r'####\s*\d+', f'\\\\boxed{{{number}}}', solution)

            formatted_item = {
                "id": idx+1,  # Use row number as ID
                "problem": item['question'],
                "type": "openai/gsm8k",  # All problems are from GSM8K
                "solution": solution,  # Use the modified solution with \boxed
            }
            formatted_data.append(formatted_item)
        # Convert to DataFrame

        # Save to a file
        output_file = "downloaded_gsm_symbolic.json"
        with open(output_file, "w") as f:
            json.dump(formatted_data, f, indent=2)

        print(f"Successfully downloaded and saved GSM8K dataset to {output_file}")
        return pd.DataFrame(formatted_data)
    except Exception as e:
        print(f"Error downloading GSM8K dataset: {e}")

if __name__ == "__main__":
    download_gsm_symbolic_dataset()

