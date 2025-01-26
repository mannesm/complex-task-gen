from loguru import logger
from pipeline.pipeline import run_pipeline
import sys
import os

from machine_learning.pipeline.models.constants import MODEL_NAMES

logger.remove()
logger.add(sys.stderr, level="INFO")
os.getenv("ENV_FILE", ".env")

if __name__ == "__main__":
    logger.info("Starting the pipeline")
    run_pipeline(model_name=MODEL_NAMES["QWEN_2_MATH_7B"], dataset_name="gsm8k")
    logger.info("Pipeline finished")