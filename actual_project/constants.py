OUTPUT_DIR = '/home/mmokkenstorm/sync/outputs'

MODEL_OUTPUT_DIR = '{OUTPUT_DIR}/models'

BASE_MATH_MODEL = 'Qwen/Qwen2.5-Math-7B-Instruct'

BASE_GSM_FINETUNE_PATH = (
    '/gpfs/home6/mmokkenstorm/sync/qwen_models/finetuned_models/math_instruct/lora'
)


prediction_patterns = [
    r'\\\(\s*\\boxed{(\d+)}\s*\\\)',
    r'\[[\s\S]*?\\boxed{(\d+)}[\s\S]*?\]',
    r'\\boxed{(\d+)}',
    '{{\d}}',
]

DATASET_LIST = {
    'gsm8k',
    'gsm_easy',
    'gsm_medium',
    'gsm_hard',
}
