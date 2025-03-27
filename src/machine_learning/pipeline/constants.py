BASIC_MATH_REASONING_PROMPT = """You are a math reasoning assistant. Your job is to solve the following problem step by step, showing your reasoning clearly and logically.

**Question**: {question}

1. Break the problem into smaller steps and explain each one as you proceed.
2. Justify each step of your reasoning, explaining why it is valid.
3. Highlight any assumptions or edge cases that may affect the solution.
4. Conclude with the final result in the following format:

#### Answer
<Provide the final solution here with concise explanation>
"""


BASIC_MATH_PROMPT = """You are a math reasoning assistant. Your job is to solve the following problem:
Question: {question}
Answer: {answer}
"""  #TODO: What is the prompt they normally use in research? Give it a question without answer: what will it generate?
#TODO: adept prompt to have same format, this will be the format of the prompt. Calculate logP of the solution part only

PERPLEXITY_THRESHOLD = 10

AUGMENTATION_PROMPT = """You are given a math word problem and its answer.
Your task is to generate a harder version of the given problem while preserving its core concept.
The harder version should increase in difficulty by adding more steps, increasing numerical complexity, or requiring deeper reasoning.

Rules for augmentation:
1. Retain the same general topic as the original
2. Increase difficulty through added complexity, not ambiguity
3. Ensure the harder question remains solvable
4. Use clear and natural wording
5. Provide a complete step-by-step solution

Original Question:
{original_question}

Original Answer:
{original_answer}

Please format your response as follows:

Harder Question:
[Your harder version of the question]

Solution:
[Step-by-step solution]

Final Answer: \boxed[numeric result]"""
VERIFIER_PROMPT = """
Prompt:
You are given a math word problem and its proposed answer. 
Your task is to verify whether the given answer is correct.
Solve the problem independently and compare your solution with the provided answer.

Rules for verification:

Solve the question step by step.
Compare your final answer with the given answer.
If the answers match, return True.
If the answers do not match, return False.
Do not assume the provided answer is correct—always verify independently.
Example Format:
Question: {question}
Proposed Answer: {proposed_answer}

Verification Result: [True/False]
"""
