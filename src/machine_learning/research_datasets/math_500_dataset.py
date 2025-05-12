import pandas as pd
from datasets import load_dataset

class Math500Dataset:
    def __init__(self, split="test"):
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)

    def to_dataframe(self):
        return pd.DataFrame(self.dataset)

def load_math_dataset():
    try:
        dataset = Math500Dataset().dataset
        problems = dataset['problem']
        solutions = dataset['solution']
        answers = dataset['answer']
        subjects = dataset['subject']
        difficulty = dataset['level']
        df = pd.DataFrame({
            'problem': problems,
            'solution': solutions,
            'answer': answers,
            'subject': subjects,
            'difficulty': difficulty
        })

        output_file = "downloaded_math_500_test.json"
        df.to_json(output_file, orient="records", lines=True)
        print(f"Successfully downloaded and saved MATH dataset to {output_file}")
        return df

    except Exception as e:
        print(f"Error downloading MATH dataset: {e}")
        return None

if __name__ == "__main__":
    df = load_math_dataset()
    if df is not None:
        print(df.head())
