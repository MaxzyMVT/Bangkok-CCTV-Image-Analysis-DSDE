"""
Find frozen camera frames in the whole dataset.

A camera "freezes" when its feed stops but the server keeps serving the last picture: the same image,
burned-in clock included, comes back with later timestamps. Counting vehicles on those frames would
repeat one moment's traffic as if it happened at every later time, so analysis must drop them.

A frame is frozen when it looks the same as the previous frame of the same camera (time order).
Live frames always differ at least by the burned-in clock, so the test is strict: both the mean and
the largest pixel difference must be tiny. The first frame of a frozen run is kept (it is real).

Output (committed):
    DataEngineering/quality/frozen_frames.csv   one row per frozen image (drop these before counting):
                                                frozen_since = time of the real first frame, run_length, diffs
    DataEngineering/quality/frozen_summary.csv  per camera: frames, frozen frames, longest run

Usage:
    python DataEngineering/quality/find_frozen_frames.py
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
from common import METADATA_CSV, find_image, preview_name, resolve_dataset_root  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
SIG_SIZE = (176, 144)        # half of 352x288: keeps the clock digits visible


def signature(path: Path):
    """Grayscale thumbnail as float32, or None when the file can't be read."""
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize(SIG_SIZE), dtype=np.float32)
    except Exception:
        return None


def flag_frozen(df: pd.DataFrame, sigs: dict, threshold: float = 0.5, max_threshold: float = 20.0) -> pd.DataFrame:
    """df: image_id, camera_id, timestamp. sigs: image_id -> array (or None).
    Adds mean_diff / max_diff to the previous frame, frozen, frozen_since (timestamp of the first copy),
    run_length (frames showing that same picture)."""
    out = df.sort_values(["camera_id", "timestamp"]).reset_index(drop=True).copy()
    mean_d, max_d, frozen, since = [], [], [], []
    prev_cam, prev_sig, run_start = None, None, None
    for r in out.itertuples():
        s = sigs.get(r.image_id)
        if r.camera_id != prev_cam or s is None or prev_sig is None or s.shape != prev_sig.shape:
            mean_d.append(np.nan); max_d.append(np.nan); is_frozen = False
        else:
            d = np.abs(s - prev_sig)
            mean_d.append(float(d.mean())); max_d.append(float(d.max()))
            is_frozen = d.mean() < threshold and d.max() < max_threshold
        if not is_frozen:
            run_start = r.timestamp
        frozen.append(is_frozen)
        since.append(run_start if is_frozen else None)
        prev_cam, prev_sig = r.camera_id, s
    out["mean_diff"], out["max_diff"], out["frozen"], out["frozen_since"] = mean_d, max_d, frozen, since

    # run_length: how many frames share the picture (the real first frame + its frozen copies)
    run_id = (~out["frozen"]).cumsum()
    out["run_length"] = out.groupby(run_id)["frozen"].transform("size")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=None, help="Folder that contains dataset/.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5, help="Max mean pixel difference (0-255) to call frozen.")
    ap.add_argument("--max-threshold", type=float, default=20.0, help="Max single-pixel difference to call frozen.")
    args = ap.parse_args()

    root = resolve_dataset_root(args.dataset_root)
    meta = pd.read_csv(METADATA_CSV, encoding="utf-8-sig", usecols=["image_id", "camera_id", "timestamp", "relative_path"])
    # one camera at a time: ~600 thumbnails (~60 MB) in memory instead of all 61k (~6 GB)
    parts = []
    with ThreadPoolExecutor(args.workers) as ex:
        for cam, g in meta.groupby("camera_id", sort=True):
            paths = [find_image(root, p) for p in g["relative_path"]]
            sigs = dict(zip(g["image_id"], ex.map(signature, paths)))
            part = flag_frozen(g.drop(columns="relative_path"), sigs, args.threshold, args.max_threshold)
            part["unreadable"] = part["image_id"].map(lambda i: sigs[i] is None)
            parts.append(part)
            print(f"  cam {cam}: {int(part['frozen'].sum())} frozen / {len(part)}", flush=True)
            del sigs
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "stem", [preview_name(c, i) for c, i in zip(out["camera_id"], out["image_id"])])
    unreadable = int(out["unreadable"].sum())
    out[out["frozen"] | out["unreadable"]].round({"mean_diff": 3, "max_diff": 1}).to_csv(
        OUT_DIR / "frozen_frames.csv", index=False,
                                                      encoding="utf-8-sig", lineterminator="\n")

    summary = (out.groupby("camera_id")
                  .agg(frames=("frozen", "size"), frozen=("frozen", "sum"), unreadable=("unreadable", "sum"),
                       longest_run=("run_length", "max"))
                  .reset_index())
    summary["frozen_pct"] = (100 * summary["frozen"] / summary["frames"]).round(1)
    summary = summary.sort_values("frozen", ascending=False)
    summary.to_csv(OUT_DIR / "frozen_summary.csv", index=False, lineterminator="\n")

    print(f"[+] {len(out)} frames, {int(out['frozen'].sum())} frozen "
          f"({100 * out['frozen'].mean():.1f} %), {unreadable} unreadable -> {OUT_DIR / 'frozen_frames.csv'}")
    print(summary[summary["frozen"] > 0].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
