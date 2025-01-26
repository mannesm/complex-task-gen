from .base_camel import BaseModel, ModelConfig

from camel.types import ModelPlatformType

from .constants import MODEL_NAMES


class Llama3Model(BaseModel):
    def __init__(self, device: str = "cpu", temperature: float = 0.5):
        """
        Qwen math model loaded with the Camel library.
        """
        config = ModelConfig(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=MODEL_NAMES["LLAMA_3_2"],
            model_config_dict={"temperature": temperature},
        )
        super().__init__(config)

    def generate_solution(self, question: str, max_new_tokens: int = 256):
        """
        Generate a solution for a math question.
        """
        prompt = f"Question: {question}\nPlease provide a detailed reasoning and the final answer."
        return super().generate_solution(prompt, max_new_tokens)
