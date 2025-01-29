from datasets import load_dataset

class GSM8KDataset:
    def __init__(self, split="test"):
        # The default configuration for gsm8k is "main"
        self.dataset = load_dataset("gsm8k", "main", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)
