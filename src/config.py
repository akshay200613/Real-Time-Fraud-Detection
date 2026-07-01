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

MASTER = "local[*]"

SHUFFLE_PARTITIONS = "8"

DRIVER_MEMORY = "8g"

EXECUTOR_MEMORY = "8g"