"""
Stationary motorcycles near shelters (footbridges / expressways) in rain vs dry.

A motorcycle box is "stationary" when the image inside it matches the camera's median background for that
2-hour window: a vehicle that stays put becomes part of the background, a passing one does not.

Usage (after recount.py and run_all.py):
    python DataAnalysis/rain/shelter.py
Writes DataAnalysis/output/shelter_share.csv, shelter_pairs.csv and DataAnalysis/figures/pair_<camera>.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from DataAnalysis.rain.common import DATA_DIR, DATASET_ROOT, FIG_DIR, OUT_DIR  # noqa: E402

MOTORCYCLE = 1
WINDOW_H = 2


def median_background(images: list) -> np.ndarray:
    return np.median(np.stack(images).astype(np.float32), axis=0)


def is_stationary(img: np.ndarray, background: np.ndarray, box_xywhn, thresh: float = 12.0) -> bool:
    h, w = img.shape[:2]
    x, y, bw, bh = box_xywhn
    x1, x2 = max(int((x - bw / 2) * w), 0), min(int(np.ceil((x + bw / 2) * w)), w)
    y1, y2 = max(int((y - bh / 2) * h), 0), min(int(np.ceil((y + bh / 2) * h)), h)
    if x2 <= x1 or y2 <= y1:
        return False
    diff = np.abs(img[y1:y2, x1:x2].astype(np.float32) - background[y1:y2, x1:x2])
    return float(diff.mean()) < thresh


def frame_stationary(counts: pd.DataFrame, boxes: pd.DataFrame, camera_ids, load_gray, window_h: int = WINDOW_H) -> pd.DataFrame:
    """Per frame of the given cameras: number of motorcycle boxes and how many of them are stationary."""
    moto = boxes[boxes["cls"] == MOTORCYCLE].groupby("image_id")
    f = counts[counts["camera_id"].isin(camera_ids)].copy()
    f["ts"] = pd.to_datetime(f["timestamp"])
    f["window"] = f["ts"].dt.floor(f"{window_h}h")
    rows = []
    for _, g in f.groupby(["camera_id", "window"]):
        imgs = {r.image_id: load_gray(r.image_id) for r in g.itertuples()}
        imgs = {k: v for k, v in imgs.items() if v is not None}
        if imgs:                               # a camera can switch resolution: keep the most common size
            sizes = pd.Series([v.shape for v in imgs.values()])
            size = sizes.mode().iloc[0]
            imgs = {k: v for k, v in imgs.items() if v.shape == size}
        if len(imgs) < 3:                      # too few frames for a background
            continue
        bg = median_background(list(imgs.values()))
        for r in g.itertuples():
            if r.image_id not in imgs:
                continue
            bs = moto.get_group(r.image_id)[["x", "y", "w", "h"]].to_numpy() if r.image_id in moto.groups else []
            still = sum(is_stationary(imgs[r.image_id], bg, b) for b in bs)
            rows.append({"image_id": r.image_id, "camera_id": r.camera_id, "timestamp": r.timestamp,
                         "moto_boxes": len(bs), "stationary_moto": still})
    return pd.DataFrame(rows, columns=["image_id", "camera_id", "timestamp", "moto_boxes", "stationary_moto"])


def _with_wet(frames: pd.DataFrame, ch: pd.DataFrame) -> pd.DataFrame:
    f = frames.assign(hour_ts=pd.to_datetime(frames["timestamp"]).dt.floor("h"))
    return f.merge(ch[["camera_id", "hour_ts", "wet"]], on=["camera_id", "hour_ts"], how="inner")


def stationary_share(frames: pd.DataFrame, ch: pd.DataFrame) -> pd.DataFrame:
    f = _with_wet(frames, ch)
    s = f.groupby(["camera_id", "wet"]).agg(moto_boxes=("moto_boxes", "sum"), stationary_moto=("stationary_moto", "sum")).reset_index()
    s["share"] = s["stationary_moto"] / s["moto_boxes"].where(s["moto_boxes"] > 0)
    return s


def pick_pairs(frames: pd.DataFrame, ch: pd.DataFrame, share: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Cameras with the biggest rise in stationary-motorcycle share in rain; for each, the dry and the wet frame
    with the most stationary motorcycles."""
    piv = share.pivot(index="camera_id", columns="wet", values="share").dropna()
    if piv.empty:
        return pd.DataFrame(columns=["camera_id", "rise", "dry_image_id", "wet_image_id"])
    rise = (piv[True] - piv[False]).sort_values(ascending=False).head(k)
    f = _with_wet(frames, ch)
    rows = []
    for cam, r in rise.items():
        g = f[f["camera_id"] == cam]
        best = lambda wet: g[g["wet"] == wet].sort_values("stationary_moto", ascending=False)["image_id"].iloc[0]
        rows.append({"camera_id": cam, "rise": r, "dry_image_id": best(False), "wet_image_id": best(True)})
    return pd.DataFrame(rows)


def main():
    import cv2
    meta = pd.read_csv(ROOT / "DataEngineering" / "cctv_images_metadata.csv", encoding="utf-8-sig",
                       usecols=["image_id", "relative_path"]).set_index("image_id")["relative_path"]
    sys.path.insert(0, str(ROOT / "DataEngineering" / "labeling"))
    from common import find_image  # noqa: E402

    def read(image_id, flag):
        return cv2.imdecode(np.fromfile(str(find_image(DATASET_ROOT, meta[image_id])), np.uint8), flag)

    counts = pd.read_csv(DATA_DIR / "counts_v2.csv")
    boxes = pd.read_csv(DATA_DIR / "boxes_v2.csv")
    cameras = pd.read_csv(OUT_DIR / "camera.csv")
    ch = pd.read_csv(OUT_DIR / "camera_hour.csv", parse_dates=["hour_ts"])
    cams = cameras.loc[cameras["near_shelter"], "camera_id"]
    wet_cams = set(ch.loc[ch["wet"], "camera_id"])
    cams = [c for c in cams if c in wet_cams]
    print(f"[*] {len(cams)} shelter cameras with rain")
    frames = frame_stationary(counts, boxes, cams, lambda i: read(i, cv2.IMREAD_GRAYSCALE))
    share = stationary_share(frames, ch)
    pairs = pick_pairs(frames, ch, share)
    share.to_csv(OUT_DIR / "shelter_share.csv", index=False, lineterminator="\n")
    pairs.to_csv(OUT_DIR / "shelter_pairs.csv", index=False, lineterminator="\n")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in pairs.itertuples():
        dry, wet = read(p.dry_image_id, cv2.IMREAD_COLOR), read(p.wet_image_id, cv2.IMREAD_COLOR)
        if dry is None or wet is None:
            continue
        side = np.hstack([dry, np.full((dry.shape[0], 8, 3), 255, np.uint8), cv2.resize(wet, (dry.shape[1], dry.shape[0]))])
        cv2.imencode(".png", side)[1].tofile(str(FIG_DIR / f"pair_{p.camera_id}.png"))
    print(f"[+] {len(pairs)} pairs -> {FIG_DIR}")


if __name__ == "__main__":
    main()
