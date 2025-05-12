from abc import ABC, abstractmethod

from datasets import load_dataset

class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, split="test"):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def to_dataframe(self):
        pass

gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")


gsm_symbolic_easy = load_dataset("apple/GSM-Symbolic", name="main")

gsm_symbolic_medium = load_dataset("apple/GSM-Symbolic", name="p1")

gsm_symbolic_hard = load_dataset("apple/GSM-Symbolic", name="p2")

math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
