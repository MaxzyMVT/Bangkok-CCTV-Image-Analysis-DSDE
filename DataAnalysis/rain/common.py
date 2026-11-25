"""Shared paths, constants, geo helpers, bootstrap and loaders for the rain insight analysis."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "DataEngineering" / "WebScraping" / "raw"
DATA_DIR = ROOT / "DataAnalysis" / "data"
OUT_DIR = ROOT / "DataAnalysis" / "output"
FIG_DIR = ROOT / "DataAnalysis" / "figures"
DATASET_ROOT = next(
    (p for p in (ROOT, ROOT.parent, Path("D:/Bangkok-CCTV-Image-Analysis-DSDE"))
     if (p / "dataset" / "100_cameras").exists()),
    ROOT,
)

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


# ---- PyDuckLake loaders & writers (Single Source of Truth) -----------------------------------------------
def _lake():
    from DataEngineering.lakehouse_pipeline import lake_has_table, query_lake, write_lake_table
    return query_lake, write_lake_table, lake_has_table


def load_rain() -> pd.DataFrame:
    query_lake, _, _ = _lake()
    r = query_lake("SELECT station_code AS gauge, CAST(datetime AS TIMESTAMP) AS hour_ts, rain_mm FROM lake.silver.drainage_rain_hourly")
    r["hour_ts"] = pd.to_datetime(r["hour_ts"])
    return r


def load_gauges() -> pd.DataFrame:
    query_lake, _, _ = _lake()
    return query_lake("""
        SELECT DISTINCT s.station_code AS gauge, s.latitude, s.longitude
        FROM lake.silver.drainage_stations s
        INNER JOIN (SELECT DISTINCT station_code FROM lake.silver.drainage_rain_hourly) r
            ON s.station_code = r.station_code
    """)


def _poi(category: str) -> pd.DataFrame:
    query_lake, _, _ = _lake()
    return query_lake("SELECT * FROM lake.silver.osm_poi WHERE category = ?", [category])


def load_rail() -> pd.DataFrame:
    return _poi("rail_station")[["name", "name_en", "latitude", "longitude"]]


def load_shelters() -> pd.DataFrame:
    """Places riders can shelter under: footbridges, and expressway points (mostly elevated in Bangkok)."""
    query_lake, _, _ = _lake()
    motorway = query_lake("SELECT latitude, longitude FROM lake.silver.osm_roads WHERE highway = 'motorway'").assign(kind="motorway")
    foot = _poi("footbridge")[["latitude", "longitude"]].assign(kind="footbridge")
    return pd.concat([foot, motorway], ignore_index=True).dropna(subset=["latitude", "longitude"])


def load_traffy() -> pd.DataFrame:
    query_lake, _, _ = _lake()
    return query_lake("SELECT * FROM lake.silver.traffy_complaints")


def load_flood() -> pd.DataFrame:
    query_lake, _, _ = _lake()
    df = query_lake("SELECT station_code, road, station, CAST(datetime AS TIMESTAMP) AS datetime, flood_cm FROM lake.silver.drainage_flood_hourly")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_camera_meta() -> pd.DataFrame:
    query_lake, _, _ = _lake()
    return query_lake("""
        SELECT TRY_CAST(camera_id AS INTEGER) AS camera_id, latitude, longitude, location_th, location_en
        FROM lake.silver.cameras
        WHERE TRY_CAST(camera_id AS INTEGER) IS NOT NULL
        ORDER BY camera_id
    """)


def load_counts() -> pd.DataFrame:
    query_lake, _, lake_has_table = _lake()
    if lake_has_table("silver", "counts_v2"):
        df = query_lake("SELECT image_id, TRY_CAST(camera_id AS INTEGER) AS camera_id, CAST(timestamp AS TIMESTAMP) AS timestamp, car, motorcycle, bus, truck, total FROM lake.silver.counts_v2")
    else:
        df = query_lake("""
            SELECT ROW_NUMBER() OVER (ORDER BY camera_id, captured_at) AS image_id,
                   TRY_CAST(camera_id AS INTEGER) AS camera_id,
                   CAST(captured_at AS TIMESTAMP) AS timestamp,
                   car, motorcycle, bus, truck, total_vehicles AS total
            FROM lake.silver.vehicle_counts
        """)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df


def load_frame_catalog() -> pd.DataFrame:
    query_lake, _, _ = _lake()
    df = query_lake("""
        SELECT TRY_CAST(image_id AS INTEGER) AS image_id,
               TRY_CAST(camera_id AS INTEGER) AS camera_id,
               filename,
               strftime(CAST(captured_at AS TIMESTAMP), '%Y-%m-%dT%H:%M:%S') AS timestamp,
               hour,
               file_size_bytes
        FROM lake.silver.cctv_frames
    """)
    df["relative_path"] = "dataset/100_cameras/cctv_images/" + df["camera_id"].astype(str) + "/" + df["filename"]
    return df


def load_gold(name: str) -> pd.DataFrame:
    query_lake, _, _ = _lake()
    return query_lake(f"SELECT * FROM lake.gold.rain_{name}")


def save_gold(name: str, df: pd.DataFrame) -> int:
    _, write_lake_table, _ = _lake()
    return write_lake_table("gold", f"rain_{name}", df, f"gold: update rain_{name}")
