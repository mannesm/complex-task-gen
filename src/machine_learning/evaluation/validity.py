# from
#
# def is_valid(problem, solution):
#     inputs = tokenizer(
#         f"Problem: {problem}\nSolution: {solution}\nIs this valid? [Yes/No]",
#         return_tensors="pt"
#     )
#     logits = verifier(**inputs).logits
#     prob_yes = torch.softmax(logits, dim=-1)[0][tokenizer("Yes").input_ids[0]]
#     return prob_yes > 0.5  # Threshold δ=0.5