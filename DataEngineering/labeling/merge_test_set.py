"""
Step 4: collect every preview a reviewer moved into review/correct/ and build the test set.

Only images judged fully correct (every vehicle boxed, right class, no extra boxes) are kept.

Outputs:
    DataEngineering/test_set/labels/<stem>.txt   committed
    DataEngineering/test_set/images/<stem>.jpg   git-ignored (copied from labeling_work)
    DataEngineering/test_set/test_manifest.csv   kept images + time/date/camera columns (committed)
    DataEngineering/test_set/review_log.csv      verdict for every candidate (committed)
    DataEngineering/test_set/dataset.yaml        `yolo val data=... split=test`

Usage:
    python DataEngineering/labeling/merge_test_set.py
    python DataEngineering/labeling/merge_test_set.py --allow-unreviewed   # merge before everyone is done
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANDIDATES_CSV, TARGET_CLASSES, TEST_SET_DIR, WORK_DIR


def validate_label(path: Path) -> list[str]:
    errors = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"{path.name}:{n} expected 5 values, got {len(parts)}")
            continue
        cls, *coords = parts
        if not cls.isdigit() or int(cls) not in TARGET_CLASSES:
            errors.append(f"{path.name}:{n} bad class {cls}")
        if any(not (0.0 <= float(v) <= 1.0) for v in coords):
            errors.append(f"{path.name}:{n} coordinate outside 0..1")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="Merge even if some previews are still unsorted.")
    args = ap.parse_args()

    cand = pd.read_csv(CANDIDATES_CSV, encoding="utf-8-sig")
    verdicts = []
    for r in cand.itertuples():
        review = WORK_DIR / r.member / "review"
        in_ok = (review / "correct" / f"{r.stem}.jpg").exists()
        in_bad = (review / "wrong" / f"{r.stem}.jpg").exists()
        if in_ok and in_bad:
            v = "conflict"
        elif in_ok:
            v = "correct"
        elif in_bad:
            v = "wrong"
        elif (review / f"{r.stem}.jpg").exists():
            v = "unreviewed"
        else:
            v = "missing"
        verdicts.append(v)
    cand["verdict"] = verdicts

    summary = pd.crosstab(cand["member"], cand["verdict"], margins=True)
    print(summary, "\n")

    if (cand["verdict"] == "conflict").any():
        sys.exit(f"[x] Same image in both correct/ and wrong/: {cand.loc[cand.verdict == 'conflict', 'stem'].tolist()}")
    pending = cand["verdict"].isin(["unreviewed", "missing"]).sum()
    if pending and not args.allow_unreviewed:
        sys.exit(f"[x] {pending} candidates not reviewed yet. Finish sorting or pass --allow-unreviewed.")

    kept = cand[cand["verdict"] == "correct"].copy()
    img_out, lbl_out = TEST_SET_DIR / "images", TEST_SET_DIR / "labels"
    for d in (img_out, lbl_out):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    errors, n_boxes = [], []
    for r in kept.itertuples():
        mdir = WORK_DIR / r.member
        lbl = mdir / "labels" / f"{r.stem}.txt"
        errors += validate_label(lbl)
        shutil.copy2(lbl, lbl_out / lbl.name)
        shutil.copy2(mdir / "images" / f"{r.stem}.jpg", img_out / f"{r.stem}.jpg")
        n_boxes.append(sum(1 for line in lbl.read_text(encoding="utf-8").splitlines() if line.strip()))
    if errors:
        sys.exit("[x] Label errors:\n  " + "\n  ".join(errors[:20]))
    kept["n_boxes"] = n_boxes

    kept.drop(columns=["verdict"]).to_csv(TEST_SET_DIR / "test_manifest.csv", index=False, encoding="utf-8-sig")
    cand[["stem", "image_id", "camera_id", "member", "time_period", "verdict"]].to_csv(
        TEST_SET_DIR / "review_log.csv", index=False, encoding="utf-8-sig")
    with open(TEST_SET_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": "images", "val": "images", "test": "images",
                        "names": TARGET_CLASSES}, f, sort_keys=False)

    reviewed = cand["verdict"].isin(["correct", "wrong"]).sum()
    print(f"[+] Test set: {len(kept)} images, {sum(n_boxes)} boxes "
          f"(YOLO fully correct on {len(kept)}/{reviewed} reviewed = {len(kept) / max(reviewed, 1):.0%})")
    print(kept.groupby("time_period")["n_boxes"].agg(images="size", boxes="sum"))
    print(f"[+] Cameras covered: {kept['camera_id'].nunique()}/{cand['camera_id'].nunique()}")


if __name__ == "__main__":
    main()
