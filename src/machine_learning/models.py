# from camel.agents import ChatAgent
# from camel.models import ModelFactory
# from camel.types import ModelPlatformType, ModelType
# from camel.configs import ChatGPTConfig
#
# # from .prompts import SYSTEM_MESSAGE_AUGMENTER_CHAIN_OF_THOUGHT
#
# gpt_4O_mini_model = ModelFactory.create(
#     model_platform=ModelPlatformType.OPENAI,
#     model_type=ModelType.GPT_4O_MINI,
#     model_config_dict=ChatGPTConfig().as_dict(),
# )
#
#
# chat_agent = ChatAgent(
#     system_message=SYSTEM_MESSAGE_AUGMENTER_CHAIN_OF_THOUGHT,
#     model=gpt_4O_mini_model,
#     message_window_size=10,
# )
#
