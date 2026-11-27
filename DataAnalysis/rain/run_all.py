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
from DataAnalysis.rain import common, effects, policy, prepare, stratified, traffy  # noqa: E402


def write_tables(tables: dict, out: Path, write_lake: bool = False) -> None:
    """Write analytical results to CSV exports (and optionally to PyDuckLake Gold tables)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t = t.copy()
        for c in t.select_dtypes("number").columns:
            t[c] = t[c].round(6 if c in ("latitude", "longitude") else 4)
        t.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
        if write_lake:
            common.save_gold(name, t)


def main():
    out = common.OUT_DIR
    counts = common.load_counts()
    meta = common.load_camera_meta()
    cams = meta[["camera_id", "latitude", "longitude"]]

    tr = common.load_traffy()
    flood_pts = traffy.flood_points(tr)
    rail = common.load_rail()
    cameras = prepare.camera_table(cams, common.load_gauges(), rail, common.load_shelters(), flood_pts)
    cameras = cameras.merge(meta[["camera_id", "location_th", "location_en"]], on="camera_id", how="left")
    ch = prepare.camera_hour(counts, cameras, common.load_rain())
    cam_eff = effects.camera_effect(ch)

    tables = {
        "camera": cameras,
        "camera_hour": ch,
        "city_effect": effects.city_effect(ch),
        "stratified_effects": stratified.stratified_effects(ch),
        "rain_timing": stratified.rain_timing(ch),
        "natural_experiment": effects.natural_experiment(ch, cameras),
        "camera_effect": cam_eff,
        "denser_share": effects.denser_share(cam_eff, cameras),
        "mealtime": effects.mealtime(ch),
        "rebound": effects.rebound(counts, ch),
        "flood_areas": traffy.flood_area_table(flood_pts),
        "flood_sensor_hours": traffy.flood_sensor_hours(common.load_flood()),
        "policy_stations": policy.policy_stations(cam_eff, ch, cameras, rail),
    }
    write_tables(tables, out, write_lake=True)

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
