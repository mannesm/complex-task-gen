import re
from loguru import logger

prediction_patterns = [
        r'(?<=#### )\d+',
    ],
reference_patterns = [
        r'(?<=#### )\d+',
    ]

def extract_numeric_value(input_string: str, regex_pattern_list: list[str]) -> int | None:

    for pattern in regex_pattern_list:
        logger.debug(f"Attempting to match pattern: {pattern} to {input_string}")
        match = re.search(pattern, input_string)
        if match:
            pred_value = int(match.group(0))
            logger.info(f"Matched prediction value: {pred_value}")
            return pred_value
    logger.info("Failed to match prediction value")
    return None

def is_equal(pred, ref):
    """
    Predicate to determine if the prediction is exactly the same as the reference.
    :param pred:
    :param ref:
    :return:
    """
    return pred.strip() == ref.strip()