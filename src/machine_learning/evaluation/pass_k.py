import pandas as pd
from scipy.special import comb  # For binomial coefficient

class PassAtKCalculator:
    def __init__(self, k):
        self.k = k

    def compute_per_question(self, df):
        """
        Computes Pass@K per unique question using the probabilistic formula.

        :param df: pandas DataFrame with columns ['question_id', 'actual_answer', 'generated_answer']
        :return: pandas DataFrame with columns ['question_id', 'pass_at_k']
        """
        results = []
        unique_questions = df["question_id"].unique()

        for question_id in unique_questions:
            question_samples = df[df["question_id"] == question_id]
            generated_answers = question_samples["generated_answer"].tolist()
            actual_answer = question_samples["actual_answer"].iloc[0]

            # Number of generated answers (n) and correct answers (c)
            n = len(generated_answers)
            c = sum(is_equal(pred, actual_answer) for pred in generated_answers)

            # If n < k, use min(n, k) in the formula
            k = min(self.k, n)

            # Compute pass@k using the formula
            if c == 0:  # No correct answers, pass@k = 0
                pass_at_k = 0.0
            elif k == 0:  # If k = 0, we can't pick any answers
                pass_at_k = 0.0
            elif c == n:  # All answers are correct, pass@k = 1
                pass_at_k = 1.0
            else:
                # Compute pass@k probability
                pass_at_k = 1 - (comb(n - c, k) / comb(n, k))

            results.append({"question_id": question_id, "pass_at_k": pass_at_k})

        return pd.DataFrame(results)

# Example equality check function (adjust as needed)
def is_equal(pred, actual):
    return pred == actual
