from .base_camel import BaseModel, ModelConfig, SupportedPlatform
from camel.types import ModelPlatformType
from .constants import MODEL_NAMES, TOKEN_MAX_DEFAULT
import logging

class QWen2Math7BModel(BaseModel):
    def __init__(self, temperature: float = 0.8, max_tokens: int = TOKEN_MAX_DEFAULT, model_platform: str = "OLLAMA"):
        """
        Qwen math model loaded with the Camel library.
        """
        if model_platform in SupportedPlatform.__members__:
            platform = SupportedPlatform[model_platform]
            logging.info(f"Qwen math model loaded with platform {platform.value}")

            if platform == SupportedPlatform.VLLM:
                model_platform_type = ModelPlatformType.VLLM
                vllm_model = MODEL_NAMES["QWEN_2_MATH_7B"]
                # Add any additional VLMM-specific logic here
                logging.info(f"VLMM model initialized: {vllm_model}")

            elif platform == SupportedPlatform.OLLAMA:
                model_platform_type = ModelPlatformType.OLLAMA
                # Add any additional OLLAMA-specific logic here
                logging.info("OLLAMA model initialized")

            else:
                raise ValueError(f"Unsupported model platform: {platform}")
        else:
            raise ValueError(f"Unsupported model platform: {model_platform}")

        # Shared configuration for both platforms
        config = ModelConfig(
            model_platform=model_platform_type,
            model_type=MODEL_NAMES["QWEN_2_MATH_7B"],
            model_config_dict={"temperature": temperature, "max_tokens": max_tokens},
        )
        super().__init__(config)