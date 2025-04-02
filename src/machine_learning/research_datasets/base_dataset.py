from abc import ABC, abstractmethod

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