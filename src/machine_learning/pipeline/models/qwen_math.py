# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
#
# class QWenMathModel:
#     def __init__(self, model_name: str):
#         """
#         Initialize the Qwen math model from Hugging Face.
#         Replace with any special logic for the 'Camel' module if needed.
#         """
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
#         self.model.eval()
#         if torch.cuda.is_available():
#             self.model.to("cuda")
#
#     def generate_solution(self, question: str, max_new_tokens: int = 256):
#         """
#         Generate a solution for a given math question using the Qwen model.
#         Return the predicted text as a string.
#         """
#         prompt = f"Question: {question}\nAnswer:"
#
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         if torch.cuda.is_available():
#             inputs = {k: v.to("cuda") for k, v in inputs.items()}
#
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs, max_new_tokens=max_new_tokens, temperature=0.2, num_return_sequences=1
#             )
#
#         prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#         # If the model's output includes the question, we can parse out the answer portion.
#         answer_part = prediction.split("Answer:")[-1].strip()
#         return answer_part
