from enum import Enum
from dataclasses import dataclass, field

import torch
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


class SupportedPlatform(Enum):
    VLLM = "VLLM"
    OLLAMA = "OLLAMA"


@dataclass
class ModelConfig:
    model_platform: ModelPlatformType
    model_type: ModelType
    device: str = "cpu"
    model_config_dict: dict = field(default_factory=dict)


class BaseModel:
    def __init__(self, config: ModelConfig, platform: SupportedPlatform = SupportedPlatform.OLLAMA):
        self.device = config.device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        if platform == SupportedPlatform.VLLM:
            model_platform = ModelPlatformType.VLLM
        elif platform == SupportedPlatform.OLLAMA:
            model_platform = ModelPlatformType.OLLAMA
        else:
            raise ValueError(f"Not supported model platform: {platform}")

        self.model = ModelFactory.create(
            model_platform=model_platform or config.model_platform,
            model_type=config.model_type,
            model_config_dict=config.model_config_dict,
        )

        self.chat_agent = ChatAgent("You are a helpful assistant.", model=self.model)

    def generate_response(self, prompt: str, return_full_response: bool = True) -> str:
        result = self.chat_agent.step(prompt)
        full_response = result.msgs[0].content
        if not return_full_response:
            return self._extract_answer(full_response)
        return full_response.strip()

    @staticmethod
    def _extract_answer(output: str) -> str:
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip()
        return output.strip()