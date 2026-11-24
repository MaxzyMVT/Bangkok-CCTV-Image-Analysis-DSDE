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
        d = ch[ch[cls + "_base"] > 0]                   # no baseline for this class -> no ratio
        d = d.assign(chg=d[cls] / d[cls + "_base"] - 1)
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


MIN_WET_HOURS = 3
FEATURES = ["near_rail", "near_shelter", "in_flood_area"]


def camera_effect(ch: pd.DataFrame, min_wet_hours: int = MIN_WET_HOURS) -> pd.DataFrame:
    """Per camera, total vehicles in wet hours vs its dry baseline; CI by resampling that camera's wet hours."""
    out = []
    for cam, g in ch[ch["wet"]].groupby("camera_id"):
        n = len(g)
        eff = pct_change(g, "total")
        if n < min_wet_hours:
            out.append({"camera_id": cam, "effect_pct": eff, "ci_low": np.nan, "ci_high": np.nan,
                        "n_wet_hours": n, "label": "too few wet hours"})
            continue
        low, high = bootstrap_ci(list(g[["total", "total_base"]].to_numpy()), _ratio_pct)
        label = "clears" if high < 0 else "denser" if low > 0 else "unaffected"
        out.append({"camera_id": cam, "effect_pct": eff, "ci_low": low, "ci_high": high, "n_wet_hours": n, "label": label})
    return pd.DataFrame(out, columns=["camera_id", "effect_pct", "ci_low", "ci_high", "n_wet_hours", "label"])


def mealtime(ch: pd.DataFrame) -> pd.DataFrame:
    """Motorcycle rain effect by hour of day; CI by resampling cameras. Meal hours flagged."""
    from DataAnalysis.rain.common import MEAL
    out = []
    for hod, g in ch[ch["wet"]].groupby("hod"):
        sums = g.groupby("camera_id")[["motorcycle", "motorcycle_base"]].sum().to_numpy()
        low, high = bootstrap_ci(list(sums), _ratio_pct)
        out.append({"hod": int(hod), "effect_pct": pct_change(g, "motorcycle"), "ci_low": low, "ci_high": high,
                    "n_cameras": len(sums), "meal": int(hod) in MEAL})
    return pd.DataFrame(out)


def denser_share(cam_effect: pd.DataFrame, cameras: pd.DataFrame, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Share of "denser" cameras with each location feature vs other labelled cameras; CI by resampling each group."""
    d = cam_effect[cam_effect["label"] != "too few wet hours"].merge(cameras[["camera_id", *FEATURES]], on="camera_id")
    den, oth = d[d["label"] == "denser"], d[d["label"] != "denser"]
    rng = np.random.default_rng(seed)
    out = []
    for f in FEATURES:
        a, b = den[f].astype(float).to_numpy(), oth[f].astype(float).to_numpy()
        diff = 100 * (a.mean() - b.mean()) if len(a) and len(b) else np.nan
        if len(a) and len(b):
            boots = [100 * (rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()) for _ in range(n)]
            low, high = np.percentile(boots, [2.5, 97.5])
        else:
            low = high = np.nan
        out.append({"feature": f, "denser_share": a.mean() if len(a) else np.nan,
                    "other_share": b.mean() if len(b) else np.nan, "diff_points": diff,
                    "ci_low": float(low), "ci_high": float(high), "n_denser": len(a), "n_other": len(b)})
    return pd.DataFrame(out)


def rebound(counts: pd.DataFrame, ch: pd.DataFrame, bins=range(-60, 121, 10)) -> pd.DataFrame:
    """Frames around each rain end (a camera's wet hour followed by a dry hour; t0 = start of the dry hour),
    as a ratio to the camera's dry baseline for that hour of day, averaged per 10-minute bin."""
    edges = list(bins)
    lo, hi, step = edges[0], edges[-1], edges[1] - edges[0]
    keys = ["camera_id", "hod", "weekend"]          # same baseline cells as prepare.add_baseline
    ch = ch.assign(weekend=pd.to_datetime(ch["hour_ts"]).dt.dayofweek >= 5)
    base = ch.groupby(keys)[["motorcycle_base", "car_base"]].first()
    wet = ch.set_index(["camera_id", "hour_ts"])["wet"]
    t0s = [(cam, h + pd.Timedelta(hours=1)) for (cam, h), w in wet.items()
           if w and (cam, h + pd.Timedelta(hours=1)) in wet.index and not wet[(cam, h + pd.Timedelta(hours=1))]]
    f = counts.assign(ts=pd.to_datetime(counts["timestamp"]))
    f["hod"] = f["ts"].dt.hour
    f["weekend"] = f["ts"].dt.dayofweek >= 5
    f = f.join(base, on=keys, how="inner")
    parts = []
    for ev, (cam, t0) in enumerate(t0s):
        g = f[f["camera_id"] == cam]
        m = (g["ts"] - t0).dt.total_seconds() / 60
        g = g[(m >= lo) & (m < hi)].assign(minute=(m[(m >= lo) & (m < hi)] // step * step).astype(int), event=ev)
        parts.append(g)
    f = pd.concat(parts) if parts else f.iloc[0:0].assign(minute=[], event=[])
    out = []
    for cls in ("motorcycle", "car"):
        r = f.assign(ratio=f[cls] / f[cls + "_base"]).replace([np.inf, -np.inf], np.nan)
        agg = r.groupby("minute").agg(ratio=("ratio", "mean"), n_events=("event", "nunique"))
        for minute in range(lo, hi, step):
            row = agg.loc[minute] if minute in agg.index else None
            out.append({"cls": cls, "minute": minute, "ratio": float(row["ratio"]) if row is not None else np.nan,
                        "n_events": int(row["n_events"]) if row is not None else 0})
    return pd.DataFrame(out)
