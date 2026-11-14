"""
Step 1: split the 100 cameras between team members and pick test-set candidate images.

Each camera gets `--per-camera` candidates spread across time_period buckets and dates,
so the test set covers rush hours, midday and night for every location.

Outputs (committed to git):
    DataEngineering/test_set/camera_assignment.csv   camera_id -> member
    DataEngineering/test_set/candidates.csv          one row per candidate image

Usage:
    python DataEngineering/labeling/sample_test_candidates.py --members A,B,C --per-camera 10
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASSIGNMENT_CSV, CANDIDATES_CSV, METADATA_CSV, TEST_SET_DIR, preview_name

PERIOD_ORDER = ["Early Morning", "Morning Rush", "Midday", "Evening Rush", "Late Evening"]


def assign_cameras(camera_ids, members, seed):
    """Shuffle cameras (seeded) and deal them round-robin so everyone gets a mix of locations."""
    cams = sorted(camera_ids)
    random.Random(seed).shuffle(cams)
    return {cam: members[i % len(members)] for i, cam in enumerate(cams)}


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
    ap.add_argument("--members", default="member_1,member_2,member_3",
                    help="Comma-separated member names (folder-safe, ASCII).")
    ap.add_argument("--per-camera", type=int, default=10, help="Candidate images per camera.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Overwrite existing candidates.csv.")
    args = ap.parse_args()

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    if CANDIDATES_CSV.exists() and not args.force:
        sys.exit(f"{CANDIDATES_CSV} already exists. Use --force to resample (this changes the test set!).")

    df = pd.read_csv(METADATA_CSV, encoding="utf-8-sig")
    assignment = assign_cameras(df["camera_id"].unique(), members, args.seed)

    rng = random.Random(args.seed)
    parts = [sample_camera(g, args.per_camera, rng) for _, g in df.groupby("camera_id", sort=True)]
    cand = pd.concat(parts).sort_values(["camera_id", "timestamp"]).reset_index(drop=True)
    cand["member"] = cand["camera_id"].map(assignment)
    cand["stem"] = [preview_name(c, i) for c, i in zip(cand["camera_id"], cand["image_id"])]

    cols = ["stem", "image_id", "camera_id", "member", "timestamp", "date", "time", "hour",
            "day_of_week", "is_weekend", "time_period", "location_en", "relative_path"]
    TEST_SET_DIR.mkdir(parents=True, exist_ok=True)
    cand[cols].to_csv(CANDIDATES_CSV, index=False, encoding="utf-8-sig")

    asg = pd.DataFrame(sorted(assignment.items()), columns=["camera_id", "member"])
    loc = df.drop_duplicates("camera_id").set_index("camera_id")["location_en"]
    asg["location_en"] = asg["camera_id"].map(loc)
    asg.to_csv(ASSIGNMENT_CSV, index=False, encoding="utf-8-sig")

    print(f"[+] {len(cand)} candidates from {cand['camera_id'].nunique()} cameras -> {CANDIDATES_CSV}")
    print(cand.groupby("member").agg(cameras=("camera_id", "nunique"), images=("stem", "size")))
    print(cand["time_period"].value_counts().reindex(PERIOD_ORDER))


if __name__ == "__main__":
    main()
