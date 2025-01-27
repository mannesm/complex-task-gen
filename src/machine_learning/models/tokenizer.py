import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.constants import MODEL_NAMES, HUGGINGFACE_MODEL_NAMES


class Tokenizer:
    def __init__(self, tokenizer_name: str = HUGGINGFACE_MODEL_NAMES["QWEN_2_MATH_7B"]):

        if tokenizer_name == MODEL_NAMES["QWEN_2_MATH_7B"]:
            self.tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAMES["QWEN_2_MATH_7B"])
            self.model = AutoModelForCausalLM.from_pretrained(HUGGINGFACE_MODEL_NAMES["QWEN_2_MATH_7B"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def tokenize_text(self, text: str) -> tuple:
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    def calculate_log_probability(self, text: str):
        # Concatenate input and output for the model's context
        tokenized_input = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokenized_input)

        logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
        probs = torch.softmax(logits, dim=-1)  # Convert to probabilities

        # Tokenize output text
        output_token_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze()

        # Extract probabilities of the output tokens
        log_probs = []
        for i, token_id in enumerate(output_token_ids):
            token_prob = probs[0, -(len(output_token_ids) - i), token_id]  # Probability of the token
            log_probs.append(torch.log(token_prob))

        return tokenized_input, log_probs, sum(log_probs).item()  # List of log probabilities and total log probability
