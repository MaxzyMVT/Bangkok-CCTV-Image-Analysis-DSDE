"""Flood-prone areas from Traffy Fondue complaints, and the DDS flood-sensor hours for case studies."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from DataAnalysis.rain.common import EARTH_R

FLOOD_WORDS = ("น้ำท่วม", "ท่วมขัง", "น้ำขัง", "ท่อตัน", "ท่อระบาย", "ระบายน้ำ")
COLS = ["latitude", "longitude", "timestamp", "description"]


def flood_complaints(traffy: pd.DataFrame) -> pd.DataFrame:
    """Complaints about flooding: the scraped is_flood flag, or keywords when the flag is missing."""
    t = traffy.dropna(subset=["latitude", "longitude"])
    if "is_flood" in t.columns:
        t = t[t["is_flood"] == 1]
    else:
        text = t["description"].fillna("").astype(str)
        t = t[text.apply(lambda s: any(w in s for w in FLOOD_WORDS))]
    return t[COLS].reset_index(drop=True)


def flood_points(traffy: pd.DataFrame, eps_m: float = 200, min_samples: int = 3) -> pd.DataFrame:
    """Flood complaints that belong to a dense cluster (DBSCAN, haversine); isolated complaints are dropped."""
    c = flood_complaints(traffy)
    if c.empty:
        return pd.DataFrame(columns=["latitude", "longitude", "cluster"])
    labels = DBSCAN(eps=eps_m / EARTH_R, min_samples=min_samples, metric="haversine").fit_predict(
        np.radians(c[["latitude", "longitude"]].to_numpy()))
    return c.assign(cluster=labels).loc[lambda d: d["cluster"] >= 0, ["latitude", "longitude", "cluster"]].reset_index(drop=True)


def flood_area_table(points: pd.DataFrame) -> pd.DataFrame:
    return (points.groupby("cluster").agg(n_complaints=("latitude", "size"), latitude=("latitude", "mean"),
                                          longitude=("longitude", "mean"))
                  .reset_index().sort_values("n_complaints", ascending=False).reset_index(drop=True))


def flood_sensor_hours(flood: pd.DataFrame, min_cm: float = 5) -> pd.DataFrame:
    """DDS road flood sensor readings >= min_cm, deepest first (sensors have no coordinates: case-study list)."""
    f = flood[flood["flood_cm"] >= min_cm].rename(columns={"datetime": "hour_ts"})
    return f[["station_code", "road", "station", "hour_ts", "flood_cm"]].sort_values("flood_cm", ascending=False).reset_index(drop=True)
