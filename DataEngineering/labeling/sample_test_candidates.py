"""
Step 1: pick test-set candidate images.

Each camera gets `--per-camera` candidates spread across time_period buckets and dates,
so the test set covers rush hours, midday and night for every location.

Output (committed to git):
    DataEngineering/test_set/candidates.csv          one row per candidate image

`core` column: the frames that must be reviewed (`--core-per-camera 3` -> 300 frames), 3 per camera in
3 different time periods. The other candidates are optional extras; reviewed extras are still used.

Usage:
    python DataEngineering/labeling/sample_test_candidates.py --per-camera 10
    python DataEngineering/labeling/sample_test_candidates.py --mark-core   # only (re)write the core column
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


def pick_core(cand: pd.DataFrame, per_camera: int = 3, seed: int = 42) -> pd.Series:
    """Boolean mask: per camera, one frame from each of `per_camera` different time periods.
    The periods rotate from camera to camera so every period gets about the same number of frames."""
    rng = random.Random(seed)
    core = pd.Series(False, index=cand.index)
    for k, (_, g) in enumerate(cand.sort_values(["camera_id", "timestamp"]).groupby("camera_id", sort=True)):
        present = [p for p in PERIOD_ORDER if p in set(g["time_period"])]
        rotated = present[k % len(present):] + present[:k % len(present)]
        chosen = []
        for p in rotated[:per_camera]:
            idx = sorted(g.index[g["time_period"] == p])
            chosen.append(idx[rng.randrange(len(idx))])
        rest = sorted(set(g.index) - set(chosen))           # camera with fewer periods: fill from the rest
        while len(chosen) < min(per_camera, len(g)):
            chosen.append(rest.pop(rng.randrange(len(rest))))
        core[chosen] = True
    return core


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-camera", type=int, default=10, help="Candidate images per camera.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    ap.add_argument("--out", type=Path, default=CANDIDATES_CSV, help="Output CSV path.")
    ap.add_argument("--core-per-camera", type=int, default=3, help="Frames per camera that must be reviewed.")
    ap.add_argument("--mark-core", action="store_true",
                    help="Keep the existing candidates and only rewrite their `core` column.")
    args = ap.parse_args()

    if args.mark_core:
        cand = pd.read_csv(args.out, encoding="utf-8-sig")
        cand["core"] = pick_core(cand, args.core_per_camera, args.seed)
        cand.to_csv(args.out, index=False, encoding="utf-8-sig", lineterminator="\n")
        print(f"[+] core: {int(cand['core'].sum())} of {len(cand)} frames -> {args.out}")
        print(cand[cand["core"]]["time_period"].value_counts().reindex(PERIOD_ORDER))
        return

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists. Use --force to rewrite it.")

    df = pd.read_csv(METADATA_CSV, encoding="utf-8-sig")

    rng = random.Random(args.seed)
    parts = [sample_camera(g, args.per_camera, rng) for _, g in df.groupby("camera_id", sort=True)]
    cand = pd.concat(parts).sort_values(["camera_id", "timestamp"]).reset_index(drop=True)
    cand["stem"] = [preview_name(c, i) for c, i in zip(cand["camera_id"], cand["image_id"])]
    cand["core"] = pick_core(cand, args.core_per_camera, args.seed)

    cols = ["stem", "image_id", "camera_id", "timestamp", "date", "time", "hour",
            "day_of_week", "is_weekend", "time_period", "location_en", "relative_path", "core"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cand[cols].to_csv(args.out, index=False, encoding="utf-8-sig", lineterminator="\n")

    print(f"[+] {len(cand)} candidates from {cand['camera_id'].nunique()} cameras -> {args.out}")
    print(cand["time_period"].value_counts().reindex(PERIOD_ORDER))


if __name__ == "__main__":
    main()
