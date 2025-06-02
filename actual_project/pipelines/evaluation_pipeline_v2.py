import logging
import re

import pandas as pd
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

SOLVER_MODEL_NAME = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
BASE_URL = 'http://localhost:8000/v1'
MAX_TOKENS_RESPONSE = 2000

client = OpenAI(base_url=BASE_URL, api_key='EMPTY')


TAG_RX = {
    'code': re.compile(r'<code>\s*```python\s*(.*?)\s*```.*?</code>', re.DOTALL | re.IGNORECASE),
    'task': re.compile(r'<task>(.*?)</task>', re.DOTALL | re.IGNORECASE),
    'solution': re.compile(r'<solution>(.*?)</solution>', re.DOTALL | re.IGNORECASE),
    'answer': re.compile(
        r'(?:<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>'  # <answer> 24.75 </answer>
        r'|####\s*([-+]?\d+(?:\.\d+)?)\b'  # #### 24.75
        r'|The final answer is[:\s]*([-+]?\d+(?:\.\d+)?)'  # The final answer is: 24.75
        r'|\\boxed\{\s*([-+]?\d+(?:\.\d+)?)(?:\\%|%)?\s*\}'  # \boxed{24.75} or \boxed{24.75\%}
        r'|\\boxed\{\s*\\frac\{(\d+)\}\{(\d+)\}\s*\})'  # \boxed{\frac{11}{20}}
        r'|The answer is[:\s]*([-+]?\d+(?:\.\d+)?)',  # The answer is 25
        re.IGNORECASE,
    ),
}

FOLDER_PREFIX = '/gpfs/home6/mmokkenstorm/augmented_datasets/'
N30_FOLDER = 'n30/'
N10_GSM8K_EVAL_FOLDER = 'n10_gsm8k_eval/'
N10_GSM8K_BASE_FOLDER = 'n10_gsm8k_base/'

SYSTEM_PROMPT = (
    'You are a helpful assistant. Solve the math problem step by step. '
    'The answer should be a number, and you should always return it in the format: '
    'The final answer is: <answer> 42 </answer>.'
)
SOLVER_PROMPT = """Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?
Let's think step by step
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30*2/3=20 kids came to the cookout.
The answer is 20

Question: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?
Let's think step by step
Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.
The total monthly earning of Zilla is represented by 100%, so $19 × 100 = $1900 is her monthly earnings.
So, $1900/2 = $950 is spent on her other monthly expenses.
The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.
Hence, she saves $1900 - $1083 = $817 per month.
The answer is 817

Question: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz’s ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?
Let's think step by step
The total ratio representing the slices of pizza that Buzz bought is 5+8=13
If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices
Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28
The answer is 28

Question: Jame gets a raise to $20 per hour and works 40 hours a week. His old job was $16 an hour for 25 hours per week. How much more money does he make per year in his new job than the old job if he works 52 weeks a year?
Let's think step by step
He makes 20*40=$800 per week
He used to make 16*25=$400 per week
So his raise was 800-400=$400 per week
So he makes 400*52=$20,800 per year more
The answer is 20800

Question: Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive?
Let's think step by step
Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats
Each student will receive 80 / 20 = 4 sweet treats
The answer is 4

Question: A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot’s vehicles in all?
Let's think step by step
The used car lot has 24 / 3 = 8 motorcycles with 2 tires each.
The lot has 24 - 8 = 16 cars for sale
There are 16 / 4 = 4 cars with a spare tire with 5 tires each.
The lot has 16 - 4 = 12 cars with 4 tires each.
Thus, the used car lot’s vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all.
The answer is 84

Question: Norma takes her clothes to the laundry. She leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds 3 sweaters and triple the number of T-shirts. How many items are missing?
Let's think step by step
Norma left 9 T-shirts
And twice as many sweaters, she took 9 * 2= 18 sweaters
Adding the T-shirts and sweaters, Norma left 9 + 18 = 27 clothes
When she came back, she found 3 sweaters
And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts
Adding the T-shirts and sweaters, Norma found 3 + 9 = 12 clothes
Subtracting the clothes she left from the clothes she found, 27 - 12 = 15 clothes are missing
The answer is 15

Question: Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard?
Let's think step by step
During 30 days Adam picked 4 * 30 = 120 apples.
So in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard.
The answer is 350

Question: {question}
Let's think step by step"""


def extract_numeric_answer(answer: str) -> float:
    match = TAG_RX['answer'].search(answer)
    if not match:
        logging.warning(f"Couldn't find answer in text:\n{answer}")
        return 0.0

    for group in match.groups():
        if group:
            try:
                return float(group)
            except ValueError:
                continue

    logging.warning("Matched but couldn't parse number")
    return 0.0


def make_prediction(question_string: str, temperature=0.8, solver_model_name: str = SOLVER_MODEL_NAME):
    response = client.chat.completions.create(
        model=solver_model_name,
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT,
            },
            {'role': 'user', 'content': SOLVER_PROMPT.format(question=question_string)},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS_RESPONSE,
    )

    completion = response.choices[0].message.content
    return completion


def evaluate_question(question: str, answer: str, k: int):
    """Evaluate a question using pass@k metric.

    Args:
        question: The math question to evaluate
        answer: The expected answer string
        k: Number of attempts/samples to make

    Returns:
        Dictionary with results including pass@k score and all predictions
    """
    expected_answer = extract_numeric_answer(answer)
    predictions = []
    correct_predictions = []
    extracted_answers = []
    # Make k predictions
    # Make k predictions
    for _ in range(k):
        try:
            completion = make_prediction(question)
            predictions.append(completion)

            # Try to extract structured <answer> tag or similar
            try:
                extracted_answer = extract_numeric_answer(completion)
                epsilon = max(1e-6, abs(expected_answer) * 0.01)
                is_correct = abs(extracted_answer - expected_answer) < epsilon
            except Exception:
                # Fallback: extract all numbers in text if parsing failed
                number_pattern = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
                answer_matches = re.findall(number_pattern, completion)
                is_correct = False
                for answer_str in reversed(answer_matches):
                    try:
                        answer_value = float(answer_str)
                        epsilon = max(1e-6, abs(expected_answer) * 0.01)
                        if abs(answer_value - expected_answer) < epsilon:
                            is_correct = True
                            break
                    except ValueError:
                        continue
            extracted_answers.append(extracted_answer)
            correct_predictions.append(is_correct)

        except Exception as e:
            logging.warning(f'Error during evaluation: {e}')
            predictions.append(f'Error: {e!s}')
            correct_predictions.append(False)

    # Calculate if passed at least once
    passed = any(correct_predictions)

    return {
        'question': question,
        'answer': answer,
        'passed': passed,
        'attempts': k,
        'model_name': SOLVER_MODEL_NAME,
        'predictions': predictions,
        'correct_predictions': correct_predictions,
        'expected_answer': expected_answer,
        'extracted_answers': extracted_answers,
        'pass_at_k': 1.0 if passed else 0.0,
        'correct_count': sum(correct_predictions),
    }


df_n30 = pd.read_csv(FOLDER_PREFIX + N30_FOLDER + 'augmented_best.csv')

# eval_result = evaluate_question(df_n30['task'].iloc[0], df_n30['solution'].iloc[0], k=5)

# eval_result_df = pd.DataFrame([eval_result])
# eval_result_df.to_csv(FOLDER_PREFIX + N30_FOLDER + 'augmented_best_eval_result.csv', index=False)


from tqdm import tqdm


def run_full_evaluation(
    df: pd.DataFrame,
    k: int = 5,
    levels: list[int] = None,
    max_samples: int = None,
) -> pd.DataFrame:
    results = []

    # Optional filtering
    if levels is not None:
        df = df[df['level'].isin(levels)]

    if max_samples is not None:
        df_unique = df.groupby('source_idx').head(1)
        if len(df_unique) >= max_samples:
            df = df_unique.sample(n=max_samples, random_state=42)
        else:
            logging.warning(f'Requested {max_samples} samples, but only {len(df_unique)} available. Using all.')
            df = df_unique

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            eval_result = evaluate_question(row['task'], row['solution'], k=k)
            eval_result.update(
                {
                    'source_idx': row['source_idx'],
                    'level': row['level'],
                    'n_augmented': df[df['source_idx'] == row['source_idx']].shape[0],
                    'code': row.get('code', ''),
                    'novelty': row.get('novelty', ''),
                    'difficulty': row.get('difficulty', ''),
                },
            )
            results.append(eval_result)
        except Exception as e:
            logging.exception(f"Failed evaluation for row {row['source_idx']}: {e}")
            continue

    return pd.DataFrame(results)


# Full evaluation at levels 0–10

result_df = run_full_evaluation(
    df_n30,
    k=5,
    levels=list(range(11)),
    max_samples=None,
)

result_df['attempts'] = 5
result_df['pass_percentage'] = result_df['correct_count'] / result_df['attempts'] * 100

# Ablation: level 3 only, 30 augmentations
# run_full_evaluation(
#     df_n30[df_n30['level'] == 3],
#     k=5,
#     max_samples=10,
#     output_path=FOLDER_PREFIX + N30_FOLDER + 'eval_ablation_level3.csv',
# )
result_df.to_csv(FOLDER_PREFIX + N30_FOLDER + 'augmented_best_eval_result.csv', index=False)
