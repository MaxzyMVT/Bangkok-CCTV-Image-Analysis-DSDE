"""Exploratory rain-response splits and a rain-timing audit.

Subgroup effects are descriptive. Different rainy clock hours can make a
weekday/weekend contrast misleading even with a matched dry baseline.
"""

import numpy as np
import pandas as pd

from DataAnalysis.rain.common import OUT_DIR, bootstrap_ci


def _effect(groups):
    a = np.asarray(groups, dtype=float)
    return float("nan") if a[:, 1].sum() == 0 else 100 * (a[:, 0].sum() / a[:, 1].sum() - 1)


def stratified_effects(ch):
    wet = ch.loc[ch["wet"]].copy()
    wet["weekend"] = wet["weekend"].astype(str).str.lower().eq("true")
    subsets = {
        "weekday": wet[~wet["weekend"]],
        "weekend": wet[wet["weekend"]],
        "light_1_to_5_mm": wet[wet["rain_mm"].between(1, 5, inclusive="left")],
        "heavy_5plus_mm": wet[wet["rain_mm"] >= 5],
    }
    rows = []
    for segment, part in subsets.items():
        for cls in ("total", "car", "motorcycle"):
            sums = part.groupby("camera_id")[[cls, cls + "_base"]].sum().to_numpy()
            if len(sums):
                effect = _effect(sums)
                lo, hi = bootstrap_ci(list(sums), _effect)
            else:
                effect = lo = hi = float("nan")
            rows.append({"segment": segment, "cls": cls, "effect_pct": effect,
                         "ci_low": lo, "ci_high": hi,
                         "n_cameras": part["camera_id"].nunique(),
                         "n_wet_hours": len(part),
                         "n_clock_hours": part["hour_ts"].nunique()})
    return pd.DataFrame(rows)


def rain_timing(ch):
    """Rain exposure and motorcycle effect by day type and hour of day."""
    wet = ch.loc[ch["wet"]].copy()
    wet["weekend"] = wet["weekend"].astype(str).str.lower().eq("true")
    wet["hod"] = pd.to_datetime(wet["hour_ts"]).dt.hour
    rows = []
    for (weekend, hod), part in wet.groupby(["weekend", "hod"]):
        sums = part[["motorcycle", "motorcycle_base"]].sum().to_numpy()
        rows.append({"day_type": "weekend" if weekend else "weekday", "hour": int(hod),
                     "n_camera_hours": len(part), "n_clock_hours": part["hour_ts"].nunique(),
                     "motorcycle_effect_pct": _effect([sums])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ch = pd.read_csv(OUT_DIR / "camera_hour.csv")
    result = stratified_effects(ch)
    result.round(4).to_csv(OUT_DIR / "stratified_effects.csv", index=False)
    rain_timing(ch).round(4).to_csv(OUT_DIR / "rain_timing.csv", index=False)
    print(result.round(2).to_string(index=False))
    print(rain_timing(ch).round(2).to_string(index=False))
