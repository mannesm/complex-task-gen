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
        pred_value, ref_value = self._extract_numeric_value(prediction=pred, reference=ref)
        return pred_value == ref_value

    @staticmethod
    def _extract_numeric_value(prediction: str, reference: str) -> tuple[int, int] | None:
        prediction_patterns = [
            r'\$\\boxed{(\d+)}',  # Matches $\\boxed{123}
            r'<<.*?=(\d+)>>',  # Matches <<...=123>>
            r'\\boxed{(\d+)}',  # Matches \\boxed{123}
            r'\\(\d+)\\',  # Matches \(123\)
            r'(\d+)',  # Matches any standalone number
            r'{(\d+)}'  # Matches any standalone number
        ]

        reference_patterns = [
            r'<<.*?=(\d+)>>',  # Matches <<...=123>>
            r'(\d+)'  # Matches any standalone number
        ]

        pred_value = None
        ref_value = None

        for pattern in prediction_patterns:
            match = re.search(pattern, prediction)
            logger.debug('Matched prediction pattern: %s', pattern)
            if match:
                pred_value = int(match.group(1))
                break

        for pattern in reference_patterns:
            match = re.search(pattern, reference)
            logger.debug('Matched reference pattern: %s', pattern)
            if match:
                ref_value = int(match.group(1))
                break

        if pred_value is not None and ref_value is not None:
            return pred_value, ref_value
        return None