from camel.types import ModelPlatformType, ModelType

from models.base_models.base_camel import BaseModel, ModelConfig


class DeepseekV3Model(BaseModel):
    def __init__(self):
        """
        SilliconFlow Deepseek V3 model loaded with the Camel library.
        """
        config = ModelConfig(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type=ModelType.SILICONFLOW_DEEPSEEK_V3,

        )
        super().__init__(config)


