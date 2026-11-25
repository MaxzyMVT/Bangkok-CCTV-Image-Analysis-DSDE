"""
Recount every frame (06:00-21:00, frozen frames excluded) with the current best detector at a fixed threshold,
keeping each box, for the rain insight analysis. DataEngineering/vehicle_counts.csv is left untouched.

Outputs: DataAnalysis/data/counts_v2.csv (one row per frame), DataAnalysis/data/boxes_v2.csv (one row per box)

Usage:
    python DataAnalysis/rain/recount.py              # full run (~30-60 min on GPU)
    python DataAnalysis/rain/recount.py --limit 20   # quick check
"""
import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from DataAnalysis.rain.common import DATA_DIR, DATASET_ROOT, HOURS, load_frame_catalog  # noqa: E402

CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
BOX_COLS = ["image_id", "cls", "x", "y", "w", "h", "conf"]


def frame_rows(result_boxes) -> dict:
    """Counts per class for one frame. result_boxes: (cls, x, y, w, h, conf) already thresholded."""
    out = {name: 0 for name in CLASSES.values()}
    for b in result_boxes:
        out[CLASSES[int(b[0])]] += 1
    out["total"] = sum(out.values())
    return out


def recount(frames: pd.DataFrame, predict: Callable[[Path], list]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts, boxes = [], []
    for r in frames.itertuples(index=False):
        found = predict(r.path)
        counts.append({"image_id": r.image_id, "camera_id": r.camera_id, "timestamp": r.timestamp, **frame_rows(found)})
        boxes += [(r.image_id, int(b[0]), *map(float, b[1:6])) for b in found]
    return pd.DataFrame(counts), pd.DataFrame(boxes, columns=BOX_COLS)


def _frames(limit: int | None) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "Models"))
    from pipeline import DEFAULT_FROZEN_CSV, load_frozen_keys  # noqa: E402
    sys.path.insert(0, str(ROOT / "DataEngineering" / "labeling"))
    from common import find_image  # noqa: E402

    meta = load_frame_catalog()
    meta = meta[meta["hour"].isin(HOURS)]
    frozen = load_frozen_keys(DEFAULT_FROZEN_CSV)
    meta = meta[[(int(c), t) not in frozen for c, t in zip(meta["camera_id"], meta["timestamp"])]]
    meta = meta.sort_values(["camera_id", "timestamp"])
    if limit:
        meta = meta.head(limit)
    meta["path"] = [find_image(DATASET_ROOT, p) for p in meta["relative_path"]]
    return meta.drop(columns=["hour", "relative_path"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.30, help="Fixed confidence threshold (chosen on the val set).")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N frames (quick check).")
    ap.add_argument("--weights", default=str(ROOT / "Models" / "weights" / "best.pt"))
    args = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(args.weights)
    device = 0 if torch.cuda.is_available() else "cpu"

    def predict(path: Path) -> list:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)   # Thai file names
        if img is None:
            return []
        b = model.predict(img, conf=args.conf, device=device, verbose=False)[0].boxes
        return [(int(c), *xywh, float(s)) for c, xywh, s in zip(b.cls.tolist(), b.xywhn.tolist(), b.conf.tolist())]

    frames = _frames(args.limit)
    missing = sum(not Path(p).exists() for p in frames["path"])
    print(f"[*] {len(frames)} frames to count ({missing} image files missing)", flush=True)
    counts, boxes = recount(frames, predict)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(DATA_DIR / "counts_v2.csv", index=False, lineterminator="\n")
    boxes.round({"x": 5, "y": 5, "w": 5, "h": 5, "conf": 4}).to_csv(DATA_DIR / "boxes_v2.csv", index=False, lineterminator="\n")
    if args.limit is None:
        from DataEngineering.lakehouse_pipeline import write_lake_table
        counts_for_lake = counts.copy()
        counts_for_lake["timestamp"] = pd.to_datetime(counts_for_lake["timestamp"])
        write_lake_table("silver", "counts_v2", counts_for_lake, "silver: update counts_v2 from recount.py")
    print(f"[+] {len(counts)} frames, {len(boxes)} boxes -> {DATA_DIR}")


if __name__ == "__main__":
    main()
