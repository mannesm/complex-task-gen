from .base_camel import BaseModel
from .constants import MODEL_NAMES
from .deepseek_r1 import DeepseekR1Model
from .llama_3_2_3B import Llama3Model
from .qwen_2_math_7b import QWen2Math7BModel

def get_model(model_name: str = "qwen_camel") -> BaseModel:
    """
    Creates and returns a model instance based on the specified model name.
    In this example, we only provide a single Qwen + Camel model.
    You can easily add more logic for different models or backends.
    """
    model_name = model_name.lower()
    if model_name == MODEL_NAMES["DEEPSEEK_R1"]:
        return DeepseekR1Model()
    if model_name == MODEL_NAMES["QWEN_2_MATH_7B"]:
        return QWen2Math7BModel()
    if model_name == MODEL_NAMES["LLAMA_3_2"]:
        return Llama3Model()
    else:
        raise ValueError(f"Unknown model name: {model_name}")