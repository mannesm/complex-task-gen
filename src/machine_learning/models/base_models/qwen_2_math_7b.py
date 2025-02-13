from .base_camel import BaseModel, ModelConfig

from camel.types import ModelPlatformType

from .constants import MODEL_NAMES, TOKEN_MAX_DEFAULT


class QWen2Math7BModel(BaseModel):
    def __init__(self, temperature: float = 0.8, max_tokens: int = TOKEN_MAX_DEFAULT):
        """
        Qwen math model loaded with the Camel library.
        """
        config = ModelConfig(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=MODEL_NAMES["QWEN_2_MATH_7B"],
            model_config_dict={"temperature": temperature, "max_tokens": max_tokens},
        )
        super().__init__(config)
