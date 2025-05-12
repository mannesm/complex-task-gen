from .base_camel import BaseModel, ModelConfig, SupportedPlatform
from camel.types import ModelPlatformType
from .constants import MODEL_NAMES, TOKEN_MAX_DEFAULT


class QWen215BModel(BaseModel):
    def __init__(self, temperature: float = 0.8, max_tokens: int = TOKEN_MAX_DEFAULT, platform: SupportedPlatform = SupportedPlatform.OLLAMA):
        """
        Qwen math model loaded with the Camel library.
        """
        model_platform = (
            ModelPlatformType.VLLM if platform == SupportedPlatform.VLLM else ModelPlatformType.OLLAMA
        )
        config = ModelConfig(
            model_platform=model_platform,
            model_type=MODEL_NAMES["QWEN_2_15B"],
            model_config_dict={"temperature": temperature, "max_tokens": max_tokens},
        )
        super().__init__(config, platform=platform)