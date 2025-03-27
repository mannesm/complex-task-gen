import re
from loguru import logger

# Corrected regex pattern for extracting boxed numbers

prediction_patterns = [
    r"\\\(\s*\\boxed{(\d+)}\s*\\\)",  # inline math mode
    r"\[[\s\S]*?\\boxed{(\d+)}[\s\S]*?\]",  # display math mode
    r"\\boxed{(\d+)}"  # standalone boxed number
]

logger.level("INFO")
def extract_numeric_value(input_string: str, regex_pattern_list: list[str]) -> int | None:
    for pattern in regex_pattern_list:
        logger.debug(f"Attempting to match pattern: {pattern}")
        match = re.search(pattern, input_string)
        if match:
            if match.lastindex:
                pred_value = int(match.group(1))
            else:
                pred_value = int(match.group(0))
            logger.debug(f"Matched prediction value: {pred_value}")
            return pred_value
    logger.info(f"Failed to match prediction value for pattern: {regex_pattern_list} to {input_string}")
    return None


def is_equal(pred: str|int, ref: str|int) -> bool:
    """
    Predicate to determine if the prediction is exactly the same as the reference.
    :param pred:
    :param ref:
    :return:
    """
    if isinstance(pred, str):
        pred.strip()
    if isinstance(ref, str):
        ref.strip()
    return pred == ref


def extract_generated_question_answer(text):
    question_match = re.search(r"Harder Question: (.*?)\nHarder Answer:", text, re.DOTALL)
    answer_match = re.search(r"Harder Answer: (.*)", text, re.DOTALL)

    question = question_match.group(1).strip() if question_match else None
    answer = answer_match.group(1).strip() if answer_match else None

    return question, answer


def check_verification_result(text):
    pattern = r"Verification Result: True"
    match = re.search(pattern, text)
    return bool(match)