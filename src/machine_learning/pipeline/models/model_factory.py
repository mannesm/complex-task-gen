from .deepseek_camel import DeepseekCamelModel

def get_model(model_name: str = "qwen_camel", device: str = "cpu"):
    """
    Creates and returns a model instance based on the specified model name.
    In this example, we only provide a single Qwen + Camel model.
    You can easily add more logic for different models or backends.
    """
    if model_name.lower() == "qwen_camel":
        return DeepseekCamelModel(device=device)
    else:
        # You can raise an error or return a default
        raise ValueError(f"Unknown model name: {model_name}")
