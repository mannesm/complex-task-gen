from .base_camel import BaseModel, ModelConfig

from camel.types import ModelPlatformType

from .constants import MODEL_NAMES


class  QWen2Math7BModel(BaseModel):
    def __init__(self, temperature: float = 0.5, max_tokens: int = 256):
        """
        Qwen math model loaded with the Camel library.
        """
        config = ModelConfig(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=MODEL_NAMES["QWEN_2_MATH_7B"],
            model_config_dict={"temperature": temperature, "max_tokens": max_tokens},
        )
        super().__init__(config)


    def generate_solution(self, question: str, max_new_tokens: int = 256):
        """
        Generate a solution for a math question.
        """
        prompt = f"Question: {question}\nPlease provide a detailed reasoning and the final answer."
        return super().generate_solution(prompt, max_new_tokens)