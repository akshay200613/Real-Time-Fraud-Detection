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

# Dynamic Spark config: use lower memory in containerized/cloud environments (e.g. Render)
IS_CLOUD = bool(os.environ.get("RENDER") or os.environ.get("CONTAINER") or os.environ.get("PORT"))

MASTER = os.environ.get("SPARK_MASTER", "local[1]" if IS_CLOUD else "local[*]")

SHUFFLE_PARTITIONS = os.environ.get("SPARK_SHUFFLE_PARTITIONS", "2" if IS_CLOUD else "8")

# PySpark requires a strict minimum of 450MB (471859200 bytes) for driver memory. 
# Render free tier provides exactly 512MB. We use 460m to satisfy PySpark while avoiding OOM.
DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "460m" if IS_CLOUD else "4g")

EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "460m" if IS_CLOUD else "4g")
