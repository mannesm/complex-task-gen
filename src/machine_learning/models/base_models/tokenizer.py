import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base_models.constants import MODEL_NAMES, HUGGINGFACE_MODEL_NAMES


class Tokenizer:
    """
    A class to handle tokenization and log probability calculation using a pre-trained model.
    """

    def __init__(self, tokenizer_name: str = HUGGINGFACE_MODEL_NAMES["QWEN_2_MATH_7B"]):
        """
        Initialize the Tokenizer with a specified tokenizer and model.

        Args:
            tokenizer_name (str): The name of the tokenizer to use. Defaults to "QWEN_2_MATH_7B".
        """
        if tokenizer_name == MODEL_NAMES["QWEN_2_MATH_7B"]:
            self.tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAMES["QWEN_2_MATH_7B"])
            self.model = AutoModelForCausalLM.from_pretrained(HUGGINGFACE_MODEL_NAMES["QWEN_2_MATH_7B"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def tokenize_text(self, text: str) -> tuple:
        """
        Tokenize the input text and convert tokens to their corresponding IDs.

        Args:
            text (str): The input text to tokenize.

        Returns:
            tuple: A tuple containing the list of tokens and their corresponding IDs.
        """
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    def calculate_log_probability(self, text: str) -> tuple:
        """
        Calculate the log probability of the input text using the instantiated model.

        Args:
            text (str): The input text for which to calculate log probabilities.

        Returns:
            tuple: A tuple containing the tokenized input, list of log probabilities for each token, and the sum of log probabilities.
        """
        tokenized_input = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = tokenized_input["input_ids"]

        with torch.no_grad():
            outputs = self.model(**tokenized_input)

        logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)  # Convert logits to log probabilities

        # Shift logits for next-token prediction
        shifted_log_probs = log_probs[:, :-1, :]  # Ignore last token logit
        target_ids = input_ids[:, 1:]  # Shift input ids for matching

        # Gather log probabilities for the actual token predictions
        token_log_probs = shifted_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        # Convert to list and sum log probabilities
        log_p_list = token_log_probs.squeeze(0).tolist()  # Convert tensor to list
        log_p_sum = sum(log_p_list)  # Sum log probabilities

        # Return tokens, log probabilities, and total log probability
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        return tokens, log_p_list, log_p_sum
