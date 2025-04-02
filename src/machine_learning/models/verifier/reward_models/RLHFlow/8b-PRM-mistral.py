from transformers import AutoModelForCausalLM, AutoTokenizer
# Source:
# https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Mistral-Data

model_name = "RLHFlow/Llama3.1-8B-PRM-Mistral-Data" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)



original_question = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
)
original_solution = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n\\boxed{72}"
augmented_question_1 = "Lena baked 36 cookies for her school event in June, and then she baked half as many cookies in July. How many cookies did Lena bake altogether in June and July?"
augmented_solution_1 = "Lena baked 36+18 = <<36+18=54>>54 cookies altogether in June and July. \boxed{54}"
augmented_question_2 = "Derek sold 60 comic books to his classmates in March, and then he sold half as many comic books in April. How many comic books did Derek sell altogether in March and April?"
augmented_solution_2 = "Derek sold 50+30 = <<50+30=80>>80 comic books altogether in March and April. \boxed{80}"


AUGMENTED_QA_JUDGE_PROMPT = """
You will be given:
1. An original question and solution pair (Original QA)
2. Two augmented question and solution pairs (Augmented QA A and Augmented QA B)

Your task is to evaluate which of the two augmented QA pairs is **more correct, relevant, and useful**, based on the original QA pair.

Please follow these criteria during your analysis:
1. **Correctness**: Does the augmented solution provide accurate and truthful information? Any misinformation or hallucination should be penalized.
2. **Relevance**: Is the augmented question and solution logically related to the original QA? The augmented QA should maintain thematic or semantic alignment.
3. **Usefulness**: Does the augmented solution add value or insight compared to the original? Is it clear and helpful?
4. **Instruction alignment**: If the original QA had implicit or explicit instructions, does the augmented version respect them?

Avoid being influenced by solution length, formatting, or position. Focus strictly on **content quality** and **faithfulness to the original QA context**.

Original QA:
Question: {original_question}
solution: {original_solution}

Augmented QA A:
Question: {augmented_question_1}
solution: {augmented_solution_1}

Augmented QA B:
Question: {augmented_question_2}
solution: {augmented_solution_2}

Please provide a step-by-step analysis of both augmented QA pairs, comparing them based on the criteria above.

If you believe Augmented QA A is better, end your analysis with '[[A]]'.
If you believe Augmented QA B is better, end your analysis with '[[B]]'.
"""

user_prompt = AUGMENTED_QA_JUDGE_PROMPT.format(
    original_question=original_question,
    original_solution=original_solution,
    augmented_question_1=augmented_question_1,
    augmented_question_2=augmented_question_2,
    augmented_solution_1=augmented_solution_1,
    augmented_solution_2=augmented_solution_2,
)
system_prompt = ""
messages = [
    {"role": "system", "content": system_prompt,},
    {"role": "user", "content": user_prompt},
]
inputs = tokenizer(user_prompt, return_tensors="pt")

# Generate the output
output = model.generate(**inputs, max_new_tokens=512)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)
