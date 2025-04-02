import pandas as pd
from datasets import load_dataset
import json

from research_datasets.base_dataset import BaseDataset


class MathDataset(BaseDataset):
    def __init__(self, split="test"):
        self.dataset = load_dataset("nlile/hendrycks-MATH-benchmark")[split]
        self.df = pd.DataFrame(self.dataset)
        self.df["id"] = self.df.index  # Add ID field if not present

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

    def __iter__(self):
        return iter(self.df.to_dict(orient="records"))

    def to_dataframe(self):
        return self.df


def download_math_dataset(number_of_problems=5000, split="train"):
    try:
        dataset = MathDataset(split=split)
        df = dataset.to_dataframe().head(number_of_problems)

        formatted_data = []
        for idx, row in df.iterrows():
            formatted_item = {
                "id": idx + 1,
                "problem": row["problem"],
                "type": "nlile/hendrycks-MATH-benchmark",
                "solution": row["solution"]
            }
            formatted_data.append(formatted_item)

        output_file = f"downloaded_math_{split}.json"
        with open(output_file, "w") as f:
            json.dump(formatted_data, f, indent=2)

        print(f"Successfully downloaded and saved MATH dataset to {output_file}")
    except Exception as e:
        print(f"Error downloading MATH dataset: {e}")


if __name__ == "__main__":
    download_math_dataset(split="train")