"""
Build every rain-insight result table from DataAnalysis/data/counts_v2.csv (run recount.py first).

Usage:
    python DataAnalysis/rain/run_all.py
Writes DataAnalysis/output/*.csv and prints a short summary. Then: python -m DataAnalysis.rain.figures
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from DataAnalysis.rain import common, effects, policy, prepare, traffy  # noqa: E402


def main():
    out = common.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(common.DATA_DIR / "counts_v2.csv")
    meta = pd.read_csv(ROOT / "DataEngineering" / "cctv_images_metadata.csv", encoding="utf-8-sig",
                       usecols=["camera_id", "latitude", "longitude"])
    cams = meta.groupby("camera_id").first().reset_index()

    tr = common.load_traffy()
    flood_pts = traffy.flood_points(tr)
    rail = common.load_rail()
    cameras = prepare.camera_table(cams, common.load_gauges(), rail, common.load_shelters(), flood_pts)
    ch = prepare.camera_hour(counts, cameras, common.load_rain())
    cam_eff = effects.camera_effect(ch)

    tables = {
        "camera": cameras,
        "camera_hour": ch,
        "city_effect": effects.city_effect(ch),
        "natural_experiment": effects.natural_experiment(ch, cameras),
        "camera_effect": cam_eff,
        "denser_share": effects.denser_share(cam_eff, cameras),
        "mealtime": effects.mealtime(ch),
        "rebound": effects.rebound(counts, ch),
        "flood_areas": traffy.flood_area_table(flood_pts),
        "flood_sensor_hours": traffy.flood_sensor_hours(common.load_flood()),
        "policy_stations": policy.policy_stations(cam_eff, ch, cameras, rail),
    }
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False, lineterminator="\n", float_format="%.4g")

    ce = tables["city_effect"].query("window == 'all'").set_index("cls")
    print(f"[+] {len(ch)} camera-hours, {int(ch['wet'].sum())} wet on {ch.loc[ch['wet'], 'camera_id'].nunique()} cameras")
    for cls in ("total", "car", "motorcycle"):
        r = ce.loc[cls]
        print(f"    rain effect {cls:<10} {r['effect_pct']:+.1f} %  (95 % CI {r['ci_low']:+.1f} .. {r['ci_high']:+.1f})")
    ne = tables["natural_experiment"].set_index("cls").loc["total"]
    print(f"    same-hour vs dry neighbours: {ne['effect_pct_points']:+.1f} pts (CI {ne['ci_low']:+.1f} .. {ne['ci_high']:+.1f}),"
          f" {int(ne['n_pairs'])} pairs in {int(ne['n_hours'])} hours")
    print("    camera labels:", cam_eff["label"].value_counts().to_dict())
    print(f"    cameras in a flood area: {int(cameras['in_flood_area'].sum())}, near rail: {int(cameras['near_rail'].sum())},"
          f" near shelter: {int(cameras['near_shelter'].sum())}")
    print(f"    stations for the policy list: {len(tables['policy_stations'])}")
    print(f"[+] tables -> {out}")


if __name__ == "__main__":
    main()
