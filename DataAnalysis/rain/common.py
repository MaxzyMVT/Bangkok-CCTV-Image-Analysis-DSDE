"""Shared paths, constants, geo helpers, bootstrap and loaders for the rain insight analysis."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "DataEngineering" / "WebScraping" / "raw"
DATA_DIR = ROOT / "DataAnalysis" / "data"
OUT_DIR = ROOT / "DataAnalysis" / "output"
FIG_DIR = ROOT / "DataAnalysis" / "figures"
DATASET_ROOT = Path("D:/Bangkok-CCTV-Image-Analysis-DSDE")   # main checkout: holds dataset/ images

WET_MM = 1.0          # rain in the hour at the camera's nearest gauge
HEAVY_MM = 5.0
HOURS = range(6, 22)  # 06:00-21:00 (hour-of-day 6..21)
RUSH = {7, 8, 9, 17, 18, 19, 20}
MEAL = {11, 12, 13, 17, 18, 19, 20}

EARTH_R = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres; broadcasts over arrays."""
    p1, l1, p2, l2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def nearest(lat, lon, others: pd.DataFrame, key: str) -> tuple[object, float]:
    d = haversine_m(lat, lon, others["latitude"].to_numpy(), others["longitude"].to_numpy())
    i = int(np.argmin(d))
    return others[key].iloc[i], float(d[i])


def within(lat, lon, points: pd.DataFrame, radius_m: float) -> bool:
    if points.empty:
        return False
    return bool((haversine_m(lat, lon, points["latitude"].to_numpy(), points["longitude"].to_numpy()) <= radius_m).any())


def bootstrap_ci(groups: list, stat, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % CI of stat(list_of_groups) when whole groups (cameras, hours) are resampled with replacement."""
    rng = np.random.default_rng(seed)
    k = len(groups)
    vals = [stat([groups[i] for i in rng.integers(0, k, k)]) for _ in range(n)]
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---- loaders ---------------------------------------------------------------------------------------------
def load_rain() -> pd.DataFrame:
    r = pd.read_csv(RAW / "bma_drainage" / "dds_rain_hourly.csv", parse_dates=["datetime"])
    return r.rename(columns={"station_code": "gauge", "datetime": "hour_ts"})[["gauge", "hour_ts", "rain_mm"]]


def load_gauges() -> pd.DataFrame:
    st = pd.read_csv(RAW / "bma_drainage" / "dds_stations.csv")
    used = set(pd.read_csv(RAW / "bma_drainage" / "dds_rain_hourly.csv", usecols=["station_code"])["station_code"])
    st = st[st["station_code"].isin(used)]
    return st.rename(columns={"station_code": "gauge"})[["gauge", "latitude", "longitude"]].reset_index(drop=True)


def _poi(category: str) -> pd.DataFrame:
    p = pd.read_csv(RAW / "openstreetmap" / "osm_poi.csv", low_memory=False)
    return p[p["category"] == category].reset_index(drop=True)


def load_rail() -> pd.DataFrame:
    return _poi("rail_station")[["name", "name_en", "latitude", "longitude"]]


def load_shelters() -> pd.DataFrame:
    """Places riders can shelter under: footbridges, and expressway points (mostly elevated in Bangkok)."""
    roads = pd.read_csv(RAW / "openstreetmap" / "osm_roads.csv", low_memory=False)
    motorway = roads.loc[roads["highway"] == "motorway", ["latitude", "longitude"]].assign(kind="motorway")
    foot = _poi("footbridge")[["latitude", "longitude"]].assign(kind="footbridge")
    return pd.concat([foot, motorway], ignore_index=True).dropna(subset=["latitude", "longitude"])


def load_traffy() -> pd.DataFrame:
    return pd.read_csv(RAW / "traffy_fondue" / "traffy_fondue.csv", low_memory=False)


def load_flood() -> pd.DataFrame:
    return pd.read_csv(RAW / "bma_drainage" / "dds_flood_hourly.csv", parse_dates=["datetime"])
