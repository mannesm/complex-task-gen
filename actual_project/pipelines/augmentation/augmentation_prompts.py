# Define the prompts

# Text-to-code prompt;
text2code_examples = """
Task: Nancy picked up 10 apples from the basket. She gave 2 apples to her friend. How many apples does Nancy have left?
Solution: Initially, Nancy had 10 apples. She gave 2 apples to her friend. So, she has 10 - 2 = 8 apples left.
Code: 
```python
def main():
    nancy_apples = 10 # Nancy initially had 10 apples
    nancy_apples -= 2 # She gave 2 apples to her friend
    print(nancy_apples)

main()
```
""".strip()

text2code_prompt = """
You will be provided a problem statement and a solution. Please, generate the code in Python that corresponds to the solution.

Examples:
{examples}

Now, please, generate the code for the following task:

Task: {task}
Solution: {solution}
""".strip()

# Augmentation prompt;
augmentation_examples = """
Task: Nancy picked up 10 apples from the basket. She gave 2 apples to her friend. How many apples does Nancy have left?

Solution: Initially, Nancy had 10 apples. 
She gave 2 apples to her friend. 
So, she has 10 - 2 = 8 apples left.
#### 8

Code: 
```python
def main():
    nancy_apples = 10 # Nancy initially had 10 apples
    nancy_apples -= 2 # She gave 2 apples to her friend
    print(nancy_apples)

main()
```

Let's add a new variable for number of oranges that Bob had.

New code: 
<code>
```python
def main():
    nancy_apples = 10 # Nancy initially had 10 apples
    bob_oranges = 3 # Bob had 3 oranges
    nancy_apples -= 2 # She gave 2 apples to her friend
    print(nancy_apples)

main()
```
</code>

New task:
<task>
Nancy picked up 10 apples from the basket. Nancy gave 2 apples to her friend. At the same time, Bob had 3 oranges. How many apples does Nancy have left?
</task>

New solution:
<solution>
Initially, Nancy had 10 apples, and Bob had 3 oranges.
Nancy gave 2 apples to her friend. 
So, she has 10 - 2 = 8 apples left. 
#### 8
</solution>
""".strip()

augmentation_prompt = """
You will be provided a problem statement, a solution, and code corresponding to the solution. Your task is to generate a new solution to the problem statement that is different from the provided solution, maximizing the novelty score that will be computed and sent back to you.

Please, first change the code, and then adapt the natural language solution and the task to the new code. Both the task, solution, and code should be consistent and sensible.
You are allowed to do the following changes:
 - Add a new variable;
 - Change the values of the variables;
 - Change the names of the variables;
 - Change the operation;
 - Change the logic steps;
 - Add a new step;
 - Remove a step;

Format your response with <code>...</code>, <task>...</task>, and <solution>...</solution> tags.

Examples:
{examples}

Now, please, create a new code, task and solution for the following:

Task: 
<task>
{task}
</task>
Solution: 
<solution>
{solution}
</solution>
Code: 
<code>
```python
{code}
```
</code>
""".strip()
