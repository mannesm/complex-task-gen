import re
import math
from loguru import logger



class PassAtK:
    def __init__(self, k=1):
        self.k = k

    def compute(self, predictions, references):
        total = len(predictions)
        correct = 0

        for pred, ref in zip(predictions, references):
            if self._is_correct(pred, ref):
                correct += 1

        return correct / total if total > 0 else math.nan
#TODO: Implement better logic ; a list of strings is returned as a predictions

    def _is_correct(self, pred, ref):
        return pred.strip() == ref.strip()

class GSM8KPassAtK(PassAtK):
    def _is_correct(self, pred, ref):
        pred_value = self._extract_numeric_value(pred)
        ref_value = self._extract_numeric_value(ref)
        return pred_value == ref_value

    @staticmethod
    def _extract_numeric_value(text):
        patterns = [
            r'\$\\boxed{(\d+)}',  # Matches $\\boxed{123}
            r'<<.*?=(\d+)>>',     # Matches <<...=123>>
            r'\\boxed{(\d+)}',    # Matches \\boxed{123}
            r'\\(\d+)\\',         # Matches \(123\)
            r'(\d+)'              # Matches any standalone number
            r'{(\d+)}'              # Matches any standalone number
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            logger.debug('Matched pattern: %s', pattern)
            if match:
                int(match.group(1))
                return int(match.group(1))
        return None
