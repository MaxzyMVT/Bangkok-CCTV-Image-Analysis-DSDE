"""Rain effects on vehicles in frame: city-wide, same-hour neighbour comparison, per camera, by hour, after rain."""
import numpy as np
import pandas as pd

from DataAnalysis.rain.common import RUSH, bootstrap_ci, haversine_m

CLASSES = ["total", "car", "motorcycle"]
WINDOWS = {"all": lambda h: np.ones(len(h), bool), "rush": lambda h: h.isin(RUSH), "other": lambda h: ~h.isin(RUSH)}


def pct_change(rows: pd.DataFrame, cls: str) -> float:
    base = rows[cls + "_base"].sum()
    return float("nan") if base == 0 else float(100 * (rows[cls].sum() / base - 1))


def _ratio_pct(groups) -> float:
    """groups: (count_sum, base_sum) pairs -> % change of the pooled sums."""
    a = np.asarray(groups, dtype=float)
    return float("nan") if a[:, 1].sum() == 0 else float(100 * (a[:, 0].sum() / a[:, 1].sum() - 1))


def city_effect(ch: pd.DataFrame) -> pd.DataFrame:
    """Wet hours vs each camera's own dry baseline, pooled; CI by resampling cameras."""
    out = []
    for window, pick in WINDOWS.items():
        wet = ch[ch["wet"] & pick(ch["hod"])]
        n_cams = wet["camera_id"].nunique()
        for cls in CLASSES:
            if not n_cams:
                out.append({"cls": cls, "window": window, "effect_pct": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                            "n_cameras": 0, "n_wet_hours": 0})
                continue
            # each camera = (sum of counts, sum of baselines) over its wet hours; resampling cameras sums these
            sums = wet.groupby("camera_id")[[cls, cls + "_base"]].sum().to_numpy()
            low, high = bootstrap_ci(list(sums), _ratio_pct)
            out.append({"cls": cls, "window": window, "effect_pct": pct_change(wet, cls), "ci_low": low,
                        "ci_high": high, "n_cameras": n_cams, "n_wet_hours": len(wet)})
    return pd.DataFrame(out)


def natural_experiment(ch: pd.DataFrame, cameras: pd.DataFrame, radius_m: float = 3000) -> pd.DataFrame:
    """Same hour: wet camera's change vs the mean change of dry (0 mm) cameras within radius_m.
    Effect in percentage points; CI by resampling wet hours."""
    pos = cameras.set_index("camera_id")[["latitude", "longitude"]]
    out = []
    for cls in CLASSES:
        d = ch.assign(chg=ch[cls] / ch[cls + "_base"] - 1)
        per_hour = []                                 # one list of pair values per wet hour
        for _, h in d.groupby("hour_ts"):
            wet, dry = h[h["wet"]], h[h["rain_mm"] == 0]
            if wet.empty or dry.empty:
                continue
            vals = []
            for w in wet.itertuples(index=False):
                dist = haversine_m(pos.at[w.camera_id, "latitude"], pos.at[w.camera_id, "longitude"],
                                   pos.loc[dry["camera_id"], "latitude"].to_numpy(),
                                   pos.loc[dry["camera_id"], "longitude"].to_numpy())
                near = dry[dist <= radius_m]
                if len(near):
                    vals.append(w.chg - near["chg"].mean())
            if vals:
                per_hour.append(vals)
        pooled = lambda hs: 100 * float(np.mean([v for h in hs for v in h])) if hs else np.nan
        low, high = bootstrap_ci(per_hour, pooled) if per_hour else (np.nan, np.nan)
        out.append({"cls": cls, "effect_pct_points": pooled(per_hour), "ci_low": low, "ci_high": high,
                    "n_hours": len(per_hour), "n_pairs": sum(len(h) for h in per_hour)})
    return pd.DataFrame(out)
