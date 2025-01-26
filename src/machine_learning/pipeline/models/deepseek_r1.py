from .base_camel import BaseModel, ModelConfig
from camel.types import ModelPlatformType, ModelType

class DeepseekR1Model(BaseModel):
    def __init__(self, device: str = "cpu"):
        """
        Qwen math model loaded with the Camel library.
        """
        config = ModelConfig(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=ModelType.DEEPSEEK_REASONER,
        )
        super().__init__(config)

    def generate_solution(self, question: str, max_new_tokens: int = 256):
        """
        Generate a solution for a math question.
        """
        prompt = f"Question: {question}\nPlease provide a detailed reasoning and the final answer."
        return super().generate_solution(prompt, max_new_tokens)
