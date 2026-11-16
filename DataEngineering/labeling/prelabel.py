"""
Run YOLO on the frames of one set and store the boxes as pre-labels for the review tool.

    DataEngineering/reviews/<set>/prelabels/<stem>.txt    cls xc yc w h conf

Safe to re-run: frames that already have a pre-label or have been reviewed are skipped.

Usage (conda activate dsde):
    python DataEngineering/labeling/prelabel.py --set test  --device 0
    python DataEngineering/labeling/prelabel.py --set train --device 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CANDIDATES_CSV, COCO_TO_TARGET, METADATA_CSV, TRAIN_MANIFEST_CSV,  # noqa: E402
                    find_image, resolve_dataset_root)
from review_store import VALID_SETS, ReviewStore  # noqa: E402


def load_items(set_name: str) -> pd.DataFrame:
    """Frames of a set: stem, image_id, camera_id, timestamp, time_period, location_en, relative_path."""
    if set_name == "test":
        return pd.read_csv(CANDIDATES_CSV, encoding="utf-8-sig")
    if set_name == "train":
        man = pd.read_csv(TRAIN_MANIFEST_CSV, encoding="utf-8-sig")
        meta = pd.read_csv(METADATA_CSV, encoding="utf-8-sig",
                           usecols=["image_id", "timestamp", "location_en", "relative_path"])
        return man.merge(meta, on="image_id", how="left")
    raise ValueError(f"unknown set: {set_name!r}")


def to_prelabel_rows(cls_coco, xywhn, conf) -> list[tuple]:
    """COCO detections -> (target_cls, xc, yc, w, h, conf); non-vehicle classes dropped."""
    return [(COCO_TO_TARGET[int(c)], *(float(v) for v in box), float(p))
            for c, box, p in zip(cls_coco, xywhn, conf) if int(c) in COCO_TO_TARGET]


def prelabel_one(store: ReviewStore, set_name: str, stem: str, rows) -> str:
    """Write a pre-label unless the frame already has one or has been reviewed."""
    if store.read_status(set_name, stem)["status"] != "pending" or store.has_prelabel(set_name, stem):
        return "skipped"
    store.write_prelabel(set_name, stem, rows)
    return "written"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=VALID_SETS)
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu'. Default: auto.")
    ap.add_argument("--dataset-root", default=None, help="Folder that contains dataset/.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N frames (for a quick check).")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    store = ReviewStore()
    items = load_items(args.set)
    if args.limit:
        items = items.head(args.limit)
    todo = [r for r in items.itertuples()
            if store.read_status(args.set, r.stem)["status"] == "pending" and not store.has_prelabel(args.set, r.stem)]
    root = resolve_dataset_root(args.dataset_root)
    print(f"[*] set={args.set}: {len(items)} frames, {len(todo)} need a pre-label | {args.model} conf={args.conf}")
    if not todo:
        return

    model = YOLO(args.model)
    written, missing = 0, []
    for r in todo:
        path = find_image(root, r.relative_path)
        data = np.fromfile(str(path), dtype=np.uint8) if path.exists() else np.array([], np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            missing.append(str(path))
            continue
        res = model.predict(img, conf=args.conf, iou=args.iou, classes=list(COCO_TO_TARGET),
                            device=args.device, verbose=False)[0]
        rows = to_prelabel_rows(res.boxes.cls.tolist(), res.boxes.xywhn.tolist(), res.boxes.conf.tolist())
        written += prelabel_one(store, args.set, r.stem, rows) == "written"
    print(f"[+] wrote {written} pre-labels -> {store.set_dir(args.set) / 'prelabels'}")
    if missing:
        print(f"[!] {len(missing)} images could not be read, e.g. {missing[0]}")


if __name__ == "__main__":
    main()
