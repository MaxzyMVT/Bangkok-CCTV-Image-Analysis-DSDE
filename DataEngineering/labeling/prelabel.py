"""
Step 2: run YOLO on the candidate images and build each member's review folder.

For every candidate this writes, under labeling_work/<member>/ (git-ignored):
    images/<stem>.jpg            original frame (ASCII file name)
    labels/<stem>.txt            YOLO-format pre-label: <cls> <xc> <yc> <w> <h>
    review/<stem>.jpg            2x preview with the boxes drawn
    review/correct/ , review/wrong/   empty folders the reviewer sorts previews into

Usage:
    python DataEngineering/labeling/prelabel.py --model yolo11x.pt --conf 0.30
    python DataEngineering/labeling/prelabel.py --members member_1      # only one person's share
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CANDIDATES_CSV, CLASS_COLORS, CLASS_LETTER, COCO_TO_TARGET, TARGET_CLASSES,
                    WORK_DIR, find_image, resolve_dataset_root)

PREVIEW_SCALE = 2


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_unicode(path: Path, img) -> None:
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise IOError(f"encode failed: {path}")
    buf.tofile(str(path))


def draw_preview(img, boxes):
    """Upscaled frame with thin class-colored boxes and a one-letter tag, plus a count bar."""
    s = PREVIEW_SCALE
    big = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for cls, x1, y1, x2, y2, _ in boxes:
        color = CLASS_COLORS[cls]
        p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
        cv2.rectangle(big, p1, p2, color, 1, cv2.LINE_AA)
        cv2.putText(big, CLASS_LETTER[cls], (p1[0] + 2, p1[1] + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
    counts = {c: sum(1 for b in boxes if b[0] == c) for c in TARGET_CLASSES}
    bar = np.full((26, big.shape[1], 3), 25, np.uint8)
    x = 6
    for c, name in TARGET_CLASSES.items():
        txt = f"{CLASS_LETTER[c]}={name}:{counts[c]}"
        cv2.putText(bar, txt, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[c], 1, cv2.LINE_AA)
        x += cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 18
    return np.vstack([bar, big])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11x.pt", help="Ultralytics weights (downloaded on first use).")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu'. Default: auto.")
    ap.add_argument("--members", default=None, help="Comma-separated subset of members to process.")
    ap.add_argument("--dataset-root", default=None, help="Folder that contains dataset/100_cameras.")
    args = ap.parse_args()

    from ultralytics import YOLO

    cand = pd.read_csv(CANDIDATES_CSV, encoding="utf-8-sig")
    if args.members:
        keep = {m.strip() for m in args.members.split(",")}
        cand = cand[cand["member"].isin(keep)]
    root = resolve_dataset_root(args.dataset_root)
    model = YOLO(args.model)
    print(f"[*] {len(cand)} images | model={args.model} conf={args.conf} | dataset root={root}")

    rows, missing = [], []
    for member, group in cand.groupby("member"):
        mdir = WORK_DIR / member
        for sub in ("images", "labels", "review/correct", "review/wrong"):
            (mdir / sub).mkdir(parents=True, exist_ok=True)
        (mdir / "classes.txt").write_text("\n".join(TARGET_CLASSES.values()) + "\n", encoding="utf-8")

        for r in group.itertuples():
            src = find_image(root, r.relative_path)
            img = imread_unicode(src) if src.exists() else None
            if img is None:
                missing.append(str(src))
                continue
            h, w = img.shape[:2]
            res = model.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                classes=list(COCO_TO_TARGET), device=args.device, verbose=False)[0]
            boxes = []
            for b in res.boxes:
                cls = COCO_TO_TARGET[int(b.cls[0])]
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                boxes.append((cls, x1, y1, x2, y2, float(b.conf[0])))

            lines = [f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                     for c, x1, y1, x2, y2, _ in boxes]
            (mdir / "labels" / f"{r.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                           encoding="utf-8")
            shutil.copy2(src, mdir / "images" / f"{r.stem}.jpg")
            # a preview already sorted into correct/ or wrong/ is left alone
            if not any((mdir / "review" / d / f"{r.stem}.jpg").exists() for d in ("correct", "wrong")):
                imwrite_unicode(mdir / "review" / f"{r.stem}.jpg", draw_preview(img, boxes))
            rows.append({"stem": r.stem, "member": member, "n_boxes": len(boxes),
                         "min_conf": round(min((b[5] for b in boxes), default=1.0), 3)})
        print(f"[+] {member}: {len(group)} images -> {mdir}")

    pd.DataFrame(rows).to_csv(WORK_DIR / "prelabel_summary.csv", index=False)
    if missing:
        print(f"[!] {len(missing)} images could not be read, e.g. {missing[0]}")
    print(f"[+] Done. Reviewers: open labeling_work/<member>/review and sort previews into correct/ or wrong/.")


if __name__ == "__main__":
    main()
