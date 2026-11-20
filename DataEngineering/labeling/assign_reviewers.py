"""
Split the frames still waiting for review between people, whole cameras at a time.

Each person gets a set of cameras (all of a camera's pending frames), so they get used to those scenes and
judge them consistently. Cameras are handed out biggest first, each to whoever has the fewest frames so far,
so the shares end up within one camera's worth of each other. The review page reads the result and shows
a "Reviewer" filter.

Output (committed): DataEngineering/reviews/<set>/assignments.csv   (stem, camera_id, reviewer)

Usage:
    python DataEngineering/labeling/assign_reviewers.py --set train --names ปัน ภัทร ครีม
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REVIEWS_DIR  # noqa: E402
from prelabel import load_items  # noqa: E402
from review_store import ReviewStore  # noqa: E402


def assign(frames: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """frames: stem, camera_id. Returns stem, camera_id, reviewer, sorted by camera then stem."""
    frames = frames[["stem", "camera_id"]].sort_values(["camera_id", "stem"]).reset_index(drop=True)
    sizes = frames.groupby("camera_id").size()
    order = sorted(sizes.index, key=lambda c: (-sizes[c], c))       # biggest cameras first
    load = {n: 0 for n in names}
    owner = {}
    for cam in order:
        who = min(names, key=lambda n: (load[n], names.index(n)))   # fewest frames so far
        owner[cam] = who
        load[who] += int(sizes[cam])
    return frames.assign(reviewer=frames["camera_id"].map(owner))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["train", "test"], default="train")
    ap.add_argument("--names", nargs="+", required=True, help="People to split the work between.")
    args = ap.parse_args()

    store = ReviewStore()
    items = load_items(args.set)
    status = store.all_status(args.set, strict=True)
    pending = items[items["stem"].map(lambda s: status.get(s, "pending") == "pending")]
    if args.set == "test" and "core" in items.columns:
        pending = pending[pending["core"].astype(bool)]              # only the frames that must be done

    out = assign(pending, args.names)
    path = Path(REVIEWS_DIR) / args.set / "assignments.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")

    def heavy(stem):  # pre-label has a bus or truck: the slower frames to check
        return any(int(b[0]) in (2, 3) for b in store.read_prelabel(args.set, stem))
    out["bus_truck"] = out["stem"].map(heavy)
    summary = out.groupby("reviewer", sort=False).agg(frames=("stem", "size"), cameras=("camera_id", "nunique"),
                                                      bus_truck_frames=("bus_truck", "sum"))
    print(f"[+] {len(out)} pending {args.set} frames -> {path}")
    print(summary.reindex(args.names).to_string())


if __name__ == "__main__":
    main()
