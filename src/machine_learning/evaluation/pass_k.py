import math
from util import is_equal


class PassAtK:
    def __init__(self, k=1):
        self.k = k

    def compute(self, predictions, references):
        total = len(predictions)
        correct = 0

        for pred, ref in zip(predictions, references):
            if is_equal(pred, ref):
                correct += 1

        return correct / total if total > 0 else math.nan


