
from dataclasses import dataclass, field

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


@dataclass
class ModelConfig:
    model_platform: ModelPlatformType
    model_type: ModelType
    device: str = "cpu"
    model_config_dict: dict = field(default_factory=dict)


class BaseModel:
    def __init__(self, config: ModelConfig):
        self.device = config.device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        self.model = ModelFactory.create(
            model_platform=config.model_platform,
            model_type=config.model_type,
            model_config_dict=config.model_config_dict,
        )

        self.chat_agent = ChatAgent("You are a helpful assistant.", model=self.model)

    def generate_response(self, prompt: str, max_new_tokens: int = 256):
        result = self.chat_agent.step(prompt)
        full_response = result.msgs[0].content
        return self._extract_answer(full_response)

    @staticmethod
    def _extract_answer(output: str) -> str:
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip()
        return output.strip()


#
# class DeepseekCamelModel(BaseCamelModel):
#     def __init__(self, device: str = "cpu"):
#         """
#         Qwen math model loaded with the Camel library.
#         """
#         super().__init__(
#             model_platform=ModelPlatformType.DEEPSEEK,
#             model_type=ModelType.DEEPSEEK_REASONER,
#             device=device,
#             model_config_dict={},  # Ensure this dictionary contains only valid keys
#         )
#
#     def generate_solution(self, question: str, max_new_tokens: int = 256):
#         """
#         Generate a solution for a math question.
#         """
#         prompt = f"Question: {question}\nPlease provide a detailed reasoning and the final answer."
#         return super().generate_solution(prompt, max_new_tokens)