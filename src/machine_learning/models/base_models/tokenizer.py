import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base_models.constants import MODEL_NAMES, HUGGINGFACE_MODEL_NAMES


class Tokenizer:
    """
    A class to handle tokenization and log probability calculation using a pre-trained model.
    """

    def __init__(self, tokenizer_name: str = HUGGINGFACE_MODEL_NAMES["QWEN_2_15B"]):
        """
        Initialize the Tokenizer with a specified tokenizer and model.

        Args:
            tokenizer_name (str): The name of the tokenizer to use. Defaults to "QWEN_2_MATH_7B".
        """
        if tokenizer_name == MODEL_NAMES["QWEN_2_MATH_7B"]:
            self.tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAMES["QWEN_2_MATH_7B"])
            self.model = AutoModelForCausalLM.from_pretrained(HUGGINGFACE_MODEL_NAMES["QWEN_2_MATH_7B"])
        if tokenizer_name == MODEL_NAMES["QWEN_2_15B"]:
            self.tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAMES["QWEN_2_15B"])
            self.model = AutoModelForCausalLM.from_pretrained(HUGGINGFACE_MODEL_NAMES["QWEN_2_15B"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
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

    def calculate_log_probability(self, text: str, prompt: str = None) -> tuple:
        """
        Calculate the log probability of the input text, optionally focusing on the part after the prompt.

        Args:
            text (str): The input text for which to calculate log probabilities.
            prompt (str, optional): The prompt part of the text. If provided, log probabilities
                                    are calculated only for the text part after the prompt.

        Returns:
            tuple: A tuple containing the solution tokens, list of log probabilities for each solution token,
                   and the sum of log probabilities.
        """
        # TODO: Update log P calculation to use the FromChatTemplate function -> Take logits from the model
        # TODO: Only for the 
        # TODO: ONly use solver to calculate logP
        # TODO: Serve augmenter using VLLM
        text, prompt = text.strip(), prompt.strip() if prompt is not None else None
        tokenized_input = self.tokenizer(text, return_tensors="pt").to(self.device)

        input_ids = tokenized_input["input_ids"]

        with torch.no_grad():
            outputs = self.model(**tokenized_input)

        logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)  # Convert logits to log probabilities

        # Shift logits for next-token prediction
        shifted_log_probs = log_probs[:, :-1, :]  # Ignore last token logit
        target_ids = input_ids[:, 1:]  # Shift input ids for next-token prediction

        # Gather log probabilities for the actual token predictions
        token_log_probs = shifted_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        # Determine start index based on prompt
        start_idx = 0
        if prompt is not None:
            # Tokenize the prompt
            prompt_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_ids = prompt_input["input_ids"].squeeze(0)
            input_ids_seq = input_ids.squeeze(0)

            # Validate prompt is at the beginning of input_ids
            if prompt_ids.shape[0] > input_ids_seq.shape[0]:
                raise ValueError("Prompt is longer than the input text.")
            if not torch.all(input_ids_seq[:len(prompt_ids)] == prompt_ids):
                raise ValueError("Input text does not start with the provided prompt.")
            start_idx = len(prompt_ids) - 1

        solution_log_probs = token_log_probs[:, start_idx:]
        log_p_list = solution_log_probs.squeeze(0).tolist()
        log_p_sum = sum(log_p_list)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        solution_tokens = tokens[start_idx + 1:]  # +1 to align with shifted log_probs

        return solution_tokens, log_p_list, log_p_sum
