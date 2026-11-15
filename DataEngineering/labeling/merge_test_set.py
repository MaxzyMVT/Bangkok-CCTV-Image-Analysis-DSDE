"""
Build the hand-verified test set from the review tool's results.

Keeps frames reviewed as `approved` (YOLO was right) or `corrected` (a person fixed the boxes);
drops `rejected` frames. Also reports how far YOLO's pre-label counts were from the reviewed counts.

Outputs in DataEngineering/test_set/:
    labels/<stem>.txt     committed
    images/<stem>.jpg     git-ignored (copied from dataset/)
    test_manifest.csv     kept frames + status
    review_log.csv        status of every candidate
    count_error.csv       YOLO pre-label count error per class and time period
    dataset.yaml          for `yolo val ... split=test`

Usage:
    python DataEngineering/labeling/merge_test_set.py
    python DataEngineering/labeling/merge_test_set.py --allow-unreviewed   # before review is finished
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TARGET_CLASSES, TEST_SET_DIR, find_image, resolve_dataset_root  # noqa: E402
from prelabel import load_items  # noqa: E402
from review_store import ReviewStore, counts  # noqa: E402

CLASS_KEYS = [*TARGET_CLASSES.values(), "total"]


def count_error_table(rows: list[dict]) -> pd.DataFrame:
    """rows: {stem, time_period, pre: counts, final: counts}. Signed error = pre - final (+ = YOLO over-counts)."""
    records = []
    for r in rows:
        for k in CLASS_KEYS:
            records.append({"class": k, "time_period": r["time_period"], "err": r["pre"][k] - r["final"][k]})
    df = pd.DataFrame(records, columns=["class", "time_period", "err"])
    df = pd.concat([df, df.assign(time_period="all")])
    out = (df.groupby(["class", "time_period"])["err"]
             .agg(n_frames="size", mae=lambda e: e.abs().mean(), mean_signed_error="mean")
             .reset_index())
    return out.round(3)


def merge(store: ReviewStore, items: pd.DataFrame, dataset_root: Path, out_dir: Path,
          allow_unreviewed: bool = False) -> dict:
    statuses = store.all_status("test", strict=True)
    items = items.copy()
    items["status"] = items["stem"].map(statuses).fillna("pending")
    tally = items["status"].value_counts().to_dict()
    print("[*] review status:", tally)
    pending = int(tally.get("pending", 0))
    if pending and not allow_unreviewed:
        sys.exit(f"[x] {pending} frames still pending. Finish reviewing or pass --allow-unreviewed.")

    kept = items[items["status"].isin(["approved", "corrected"])].copy()
    img_out, lbl_out = Path(out_dir) / "images", Path(out_dir) / "labels"
    for d in (img_out, lbl_out):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    rows, n_boxes, missing = [], [], []
    for r in kept.itertuples():
        boxes = store.read_labels("test", r.stem) or []
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
        (lbl_out / f"{r.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        src = find_image(Path(dataset_root), r.relative_path)
        if src.exists():
            shutil.copy2(src, img_out / f"{r.stem}.jpg")
        else:
            missing.append(str(src))
        n_boxes.append(len(boxes))
        rows.append({"stem": r.stem, "time_period": r.time_period,
                     "pre": counts(store.read_prelabel("test", r.stem)), "final": counts(boxes)})
    kept["n_boxes"] = n_boxes

    out_dir = Path(out_dir)
    kept.to_csv(out_dir / "test_manifest.csv", index=False, encoding="utf-8-sig")
    items[["stem", "image_id", "camera_id", "time_period", "status"]].to_csv(
        out_dir / "review_log.csv", index=False, encoding="utf-8-sig")
    errors = count_error_table(rows)
    errors.to_csv(out_dir / "count_error.csv", index=False)
    with open(out_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": "images", "val": "images", "test": "images", "names": TARGET_CLASSES},
                       f, sort_keys=False)

    print(f"[+] test set: {len(kept)} frames, {sum(n_boxes)} boxes -> {out_dir}")
    if rows:
        print("[+] YOLO pre-label count error (all time periods):")
        print(errors[errors["time_period"] == "all"].to_string(index=False))
    if missing:
        print(f"[!] {len(missing)} images missing from dataset/, e.g. {missing[0]}")
    return {"kept": len(kept), "boxes": sum(n_boxes), "status": tally, "missing_images": len(missing)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-unreviewed", action="store_true", help="Merge even if some frames are still pending.")
    ap.add_argument("--dataset-root", default=None, help="Folder that contains dataset/.")
    args = ap.parse_args()
    merge(ReviewStore(), load_items("test"), resolve_dataset_root(args.dataset_root), TEST_SET_DIR,
          allow_unreviewed=args.allow_unreviewed)


if __name__ == "__main__":
    main()
