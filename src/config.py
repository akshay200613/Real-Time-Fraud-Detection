import os

# ===================================
# PROJECT PATHS
# ===================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(BASE_DIR, "data", "raw")

PROCESSED_DATA = os.path.join(BASE_DIR, "data", "processed")

PARQUET_DATA = os.path.join(BASE_DIR, "data", "parquet")

MODEL_PATH = os.path.join(BASE_DIR, "models")

# ===================================
# SPARK CONFIGURATION
# ===================================

APP_NAME = "Real-Time Fraud Detection"

# Dynamic Spark config: use cloud defaults suited for Hugging Face Spaces (16GB RAM)
IS_CLOUD = bool(os.environ.get("RENDER") or os.environ.get("CONTAINER") or os.environ.get("PORT") or os.environ.get("SPACE_ID"))

MASTER = os.environ.get("SPARK_MASTER", "local[*]")

SHUFFLE_PARTITIONS = os.environ.get("SPARK_SHUFFLE_PARTITIONS", "4" if IS_CLOUD else "8")

DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "2g" if IS_CLOUD else "4g")

EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "2g" if IS_CLOUD else "4g")
