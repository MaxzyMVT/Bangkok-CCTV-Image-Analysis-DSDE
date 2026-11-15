"""
Step 1: pick test-set candidate images.

Each camera gets `--per-camera` candidates spread across time_period buckets and dates,
so the test set covers rush hours, midday and night for every location.

Output (committed to git):
    DataEngineering/test_set/candidates.csv          one row per candidate image

Usage:
    python DataEngineering/labeling/sample_test_candidates.py --per-camera 10
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANDIDATES_CSV, METADATA_CSV, preview_name

PERIOD_ORDER = ["Early Morning", "Morning Rush", "Midday", "Evening Rush", "Late Evening"]


def sample_camera(df_cam: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    """Pick n images spread evenly over time periods, and over dates within each period."""
    periods = [p for p in PERIOD_ORDER if p in set(df_cam["time_period"])]
    # quota per period, distributing the remainder to the busiest periods first
    quota = {p: n // len(periods) for p in periods}
    by_size = sorted(periods, key=lambda p: -int((df_cam["time_period"] == p).sum()))
    for p in by_size[: n - sum(quota.values())]:
        quota[p] += 1

    picked = []
    for p in periods:
        pool = df_cam[df_cam["time_period"] == p]
        dates = sorted(pool["date"].unique())
        rng.shuffle(dates)
        chosen_ids = set()
        # one image per date in turn, so the picks land on different days
        while len(chosen_ids) < min(quota[p], len(pool)):
            for d in dates:
                if len(chosen_ids) >= quota[p]:
                    break
                remaining = pool[(pool["date"] == d) & (~pool["image_id"].isin(chosen_ids))]
                if remaining.empty:
                    continue
                chosen_ids.add(int(remaining.iloc[rng.randrange(len(remaining))]["image_id"]))
        picked.append(pool[pool["image_id"].isin(chosen_ids)])
    return pd.concat(picked)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-camera", type=int, default=10, help="Candidate images per camera.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    ap.add_argument("--out", type=Path, default=CANDIDATES_CSV, help="Output CSV path.")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists. Use --force to rewrite it.")

    df = pd.read_csv(METADATA_CSV, encoding="utf-8-sig")

    rng = random.Random(args.seed)
    parts = [sample_camera(g, args.per_camera, rng) for _, g in df.groupby("camera_id", sort=True)]
    cand = pd.concat(parts).sort_values(["camera_id", "timestamp"]).reset_index(drop=True)
    cand["stem"] = [preview_name(c, i) for c, i in zip(cand["camera_id"], cand["image_id"])]

    cols = ["stem", "image_id", "camera_id", "timestamp", "date", "time", "hour",
            "day_of_week", "is_weekend", "time_period", "location_en", "relative_path"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cand[cols].to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[+] {len(cand)} candidates from {cand['camera_id'].nunique()} cameras -> {args.out}")
    print(cand["time_period"].value_counts().reindex(PERIOD_ORDER))


if __name__ == "__main__":
    main()
