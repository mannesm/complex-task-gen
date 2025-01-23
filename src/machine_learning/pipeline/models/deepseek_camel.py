import torch
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


def _extract_answer(output: str) -> str:
    """
    Simple method to parse or clean up the raw answer from Camel’s output.
    """
    if "Answer:" in output:
        return output.split("Answer:")[-1].strip()
    return output.strip()


class DeepseekCamelModel:
    def __init__(self, device="cpu"):
        """
        Example of a Qwen math model loaded with the Camel library.
        Adjust the code based on how Camel can load or chat with Qwen.
        """
        # Create the model using ModelFactory with valid configuration

        self.device = device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_REASONER,
            model_config_dict={},  # Ensure this dictionary contains only valid keys
            # model_config_dict={"device":self.device},  #!TODO: Check if this is correct, use this to host local?
        )

        # Initialize ChatAgent with the created model
        self.chat_agent = ChatAgent("You are a helpful math solver.", model=self.model)

    def generate_solution(self, question: str, max_new_tokens: int = 256):
        """
        Generate a solution for a math question.
        This method uses the Camel ChatAgent to produce an answer.
        """
        prompt = f"Question: {question}\nPlease provide a detailed reasoning and the final answer."
        result = self.chat_agent.step(
            prompt,
        )
        full_response = result.msgs[0].content
        answer_part = _extract_answer(full_response)
        return answer_part