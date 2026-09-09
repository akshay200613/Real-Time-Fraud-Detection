"""
metrics_store.py

Persists model evaluation results to disk so the API can serve them
without requiring the dashboard to hardcode any values.
"""

import json
import os
from datetime import datetime, timezone
from src.utils.logger import logger


METRICS_PATH = os.path.join("models", "metrics.json")


def save_metrics(results: list, path: str = METRICS_PATH) -> None:
    """
    Saves evaluation results to a JSON file.

    Args:
        results: List of per-model evaluation dicts from ModelEvaluator.evaluate()
        path:    Output file path (default: models/metrics.json)
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    entry = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "models": results
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)

    logger.info(f"Metrics saved to {path}")


def load_metrics(path: str = METRICS_PATH) -> dict:
    """
    Loads the most recently saved metrics from disk.

    Returns:
        Dict with keys: trained_at, models
        Returns None if file does not exist.
    """

    if not os.path.exists(path):
        logger.warning(f"Metrics file not found at {path}. Run main.py to train.")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
