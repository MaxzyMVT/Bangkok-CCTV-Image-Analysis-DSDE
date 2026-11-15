"""
Which YOLO weights the App / batch pipeline should use.

Priority:
    1. explicit argument (e.g. --model)
    2. environment variable CCTV_MODEL
    3. Models/weights/best.pt  (written by Models/training/train_pipeline.py)
    4. pretrained COCO fallback (yolov8n.pt)
"""
import json
import os
from pathlib import Path
from typing import Optional

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
BEST_MODEL = WEIGHTS_DIR / "best.pt"
BEST_MODEL_INFO = WEIGHTS_DIR / "best.json"
PRETRAINED_FALLBACK = "yolov8n.pt"


def resolve_model(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("CCTV_MODEL")
    if env:
        return env
    if BEST_MODEL.exists():
        return str(BEST_MODEL)
    return PRETRAINED_FALLBACK


def best_model_info() -> dict:
    """Metadata saved next to best.pt (training run, metrics, date). Empty if none."""
    try:
        return json.loads(BEST_MODEL_INFO.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
