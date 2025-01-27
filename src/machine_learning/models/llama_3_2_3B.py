from .base_camel import BaseModel, ModelConfig

from camel.types import ModelPlatformType

from .constants import MODEL_NAMES, TOKEN_MAX_DEFAULT


class Llama3Model(BaseModel):
    def __init__(self, device: str = "cpu", temperature: float = 0.5, max_tokens: int = TOKEN_MAX_DEFAULT):
        """
        Qwen math model loaded with the Camel library.
        """
        config = ModelConfig(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=MODEL_NAMES["LLAMA_3_2"],
            model_config_dict={"temperature": temperature, "max_tokens": TOKEN_MAX_DEFAULT},
        )
        super().__init__(config)