"""
Storage for the review tool. Owns everything under DataEngineering/reviews/<set>/:

    prelabels/<stem>.txt   YOLO pre-label, "cls xc yc w h conf"  (written once, never edited)
    labels/<stem>.txt      reviewed label, "cls xc yc w h"       (written on save)
    status/<stem>.json     {"status", "reviewed_at", "n_boxes"}  (no file = pending)

One file per frame, so people reviewing different frames never conflict in git.
"""
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from common import REVIEWS_DIR, TARGET_CLASSES

Box = tuple[int, float, float, float, float]  # cls, xc, yc, w, h (normalised 0..1)
VALID_SETS = ("test", "train")
STEM_RE = re.compile(r"^cam\d{4}_img\d{5}$")
MIN_SIZE = 0.005   # boxes thinner than this (≈2 px on a 352 px frame) are dropped
IOU_MIN = 0.9


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        # On Windows, replace fails while another request is reading the target: retry briefly.
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def clean_boxes(boxes) -> list[Box]:
    """Clip each box to the frame and drop near-zero ones."""
    out = []
    for b in boxes:
        cls, xc, yc, w, h = int(b[0]), *(float(v) for v in b[1:5])
        x1, x2 = max(0.0, xc - w / 2), min(1.0, xc + w / 2)
        y1, y2 = max(0.0, yc - h / 2), min(1.0, yc + h / 2)
        if cls not in TARGET_CLASSES or x2 - x1 < MIN_SIZE or y2 - y1 < MIN_SIZE:
            continue
        out.append((cls, round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6), round(x2 - x1, 6), round(y2 - y1, 6)))
    return out


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


OVERLAP_IOU = 0.6


def overlap_pairs(boxes, iou_min: float = OVERLAP_IOU) -> list[tuple[int, int]]:
    """Index pairs of boxes that overlap so much they are almost surely one vehicle boxed twice
    (YOLO keeps a car box and a truck box on the same pickup because it suppresses per class)."""
    return [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
            if _iou(boxes[i], boxes[j]) >= iou_min]


def same_as_prelabel(pre: list[Box], boxes: list[Box], iou_min: float = IOU_MIN) -> bool:
    """True when every box matches a pre-label box of the same class with IoU >= iou_min (one-to-one)."""
    if len(pre) != len(boxes):
        return False
    pairs = sorted(((_iou(p, b), i, j) for i, p in enumerate(pre) for j, b in enumerate(boxes) if p[0] == b[0]),
                   reverse=True)
    used_p, used_b = set(), set()
    for iou, i, j in pairs:
        if iou < iou_min:
            break
        if i not in used_p and j not in used_b:
            used_p.add(i)
            used_b.add(j)
    return len(used_p) == len(pre)


def counts(boxes) -> dict[str, int]:
    out = {name: 0 for name in TARGET_CLASSES.values()}
    for b in boxes:
        out[TARGET_CLASSES[int(b[0])]] += 1
    out["total"] = sum(out.values())
    return out


def _parse(text: str, n_cols: int) -> list[tuple]:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append((int(parts[0]), *(float(v) for v in parts[1:n_cols])))
    return rows


class ReviewStore:
    def __init__(self, root: Path = REVIEWS_DIR):
        self.root = Path(root)

    def set_dir(self, set_name: str) -> Path:
        if set_name not in VALID_SETS:
            raise ValueError(f"unknown set: {set_name!r}")
        return self.root / set_name

    def _path(self, set_name: str, kind: str, stem: str) -> Path:
        if not STEM_RE.match(stem or ""):
            raise ValueError(f"bad stem: {stem!r}")
        ext = "json" if kind == "status" else "txt"
        return self.set_dir(set_name) / kind / f"{stem}.{ext}"

    # pre-labels
    def read_prelabel(self, set_name, stem) -> list[tuple]:
        p = self._path(set_name, "prelabels", stem)
        return _parse(p.read_text(encoding="utf-8"), 6) if p.exists() else []

    def has_prelabel(self, set_name, stem) -> bool:
        return self._path(set_name, "prelabels", stem).exists()

    def write_prelabel(self, set_name, stem, boxes_with_conf, overwrite=False) -> bool:
        p = self._path(set_name, "prelabels", stem)
        if p.exists() and not overwrite:
            return False
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}" for c, x, y, w, h, conf in boxes_with_conf]
        _atomic_write(p, "\n".join(lines) + ("\n" if lines else ""))
        return True

    # reviewed labels + status
    def read_labels(self, set_name, stem) -> list[Box] | None:
        p = self._path(set_name, "labels", stem)
        return _parse(p.read_text(encoding="utf-8"), 5) if p.exists() else None

    def read_status(self, set_name, stem) -> dict:
        p = self._path(set_name, "status", stem)
        if not p.exists():
            return {"status": "pending"}
        return json.loads(p.read_text(encoding="utf-8"))

    def current_boxes(self, set_name, stem) -> list[Box]:
        saved = self.read_labels(set_name, stem)
        return saved if saved is not None else [tuple(b[:5]) for b in self.read_prelabel(set_name, stem)]

    def all_status(self, set_name, strict: bool = False) -> dict[str, str]:
        """stem -> status for every saved frame. A broken file (e.g. git conflict markers) is skipped
        with a warning, or raises ValueError naming the file when strict=True (merge/build)."""
        out = {}
        for p in (self.set_dir(set_name) / "status").glob("*.json"):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))["status"]
            except (ValueError, KeyError, TypeError) as e:
                if strict:
                    raise ValueError(f"broken status file {p}: {e}. Fix or delete it, then re-run.") from e
                print(f"[!] skipping broken status file {p} ({e}); the frame shows as pending")
        return out

    def save(self, set_name, stem, boxes, action: str) -> dict:
        label_p = self._path(set_name, "labels", stem)
        status_p = self._path(set_name, "status", stem)
        if action == "reset":
            label_p.unlink(missing_ok=True)
            status_p.unlink(missing_ok=True)
            return {"status": "pending"}
        if action == "reject":
            label_p.unlink(missing_ok=True)
            status = {"status": "rejected", "n_boxes": 0}
        elif action == "save":
            boxes = clean_boxes(boxes)
            pre = [tuple(b[:5]) for b in self.read_prelabel(set_name, stem)]
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
            _atomic_write(label_p, "\n".join(lines) + ("\n" if lines else ""))
            status = {"status": "approved" if same_as_prelabel(pre, boxes) else "corrected", "n_boxes": len(boxes)}
        else:
            raise ValueError(f"unknown action: {action!r}")
        status["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_write(status_p, json.dumps(status))
        return status
