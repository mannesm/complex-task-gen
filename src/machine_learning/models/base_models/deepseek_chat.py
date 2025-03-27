from camel.types import ModelPlatformType, ModelType

from models.base_models.base_camel import BaseModel, ModelConfig


class DeepseekChatModel(BaseModel):
    def __init__(self):
        """
        Qwen math model loaded with the Camel library.
        """
        config = ModelConfig(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,

        )
        super().__init__(config)
