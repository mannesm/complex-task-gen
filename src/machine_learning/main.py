from loguru import logger
from pipeline.pipeline import run_pipeline
import sys
import os


logger.remove()
logger.add(sys.stderr, level="INFO")
os.getenv("ENV_FILE", ".env")

if __name__ == "__main__":
    logger.info("Starting the pipeline")
    run_pipeline()
    logger.info("Pipeline finished")