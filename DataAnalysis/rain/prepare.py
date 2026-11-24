"""Camera table (location features) and camera-hour table (hourly counts, rain at the nearest gauge, dry baseline)."""
import pandas as pd

from DataAnalysis.rain.common import HEAVY_MM, HOURS, WET_MM, haversine_m, nearest, within

RAIL_NEAR_M = 500
SHELTER_NEAR_M = 150
FLOOD_NEAR_M = 100
COUNT_COLS = ["car", "motorcycle", "bus", "truck", "total"]
BASE_COLS = ["car", "motorcycle", "total"]


def camera_table(cams: pd.DataFrame, gauges: pd.DataFrame, rail: pd.DataFrame, shelters: pd.DataFrame,
                 flood_pts: pd.DataFrame | None = None) -> pd.DataFrame:
    flood_pts = flood_pts if flood_pts is not None else pd.DataFrame(columns=["latitude", "longitude"])
    rows = []
    for c in cams.itertuples(index=False):
        gauge, gdist = nearest(c.latitude, c.longitude, gauges, "gauge")
        rail_d = float(haversine_m(c.latitude, c.longitude, rail["latitude"], rail["longitude"]).min()) if len(rail) else float("inf")
        rows.append({"camera_id": c.camera_id, "latitude": c.latitude, "longitude": c.longitude,
                     "gauge": gauge, "gauge_dist_m": round(gdist), "rail_dist_m": round(rail_d),
                     "near_rail": rail_d <= RAIL_NEAR_M,
                     "near_shelter": within(c.latitude, c.longitude, shelters, SHELTER_NEAR_M),
                     "in_flood_area": within(c.latitude, c.longitude, flood_pts, FLOOD_NEAR_M)})
    return pd.DataFrame(rows)


def add_baseline(ch: pd.DataFrame) -> pd.DataFrame:
    """<cls>_base = mean over the same camera's dry hours at the same hour of day. Rows with no baseline are dropped."""
    base = (ch[~ch["wet"]].groupby(["camera_id", "hod"])[BASE_COLS].mean().add_suffix("_base").reset_index())
    out = ch.merge(base, on=["camera_id", "hod"], how="inner")
    return out[out["total_base"] > 0].reset_index(drop=True)


def camera_hour(counts: pd.DataFrame, cameras: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    c = counts.copy()
    c["hour_ts"] = pd.to_datetime(c["timestamp"]).dt.floor("h")
    ch = (c.groupby(["camera_id", "hour_ts"])
            .agg(n_frames=("total", "size"), **{k: (k, "mean") for k in COUNT_COLS})
            .reset_index())
    ch["hod"] = ch["hour_ts"].dt.hour
    ch = ch[ch["hod"].isin(HOURS)]
    ch["weekend"] = ch["hour_ts"].dt.dayofweek >= 5
    ch = ch.merge(cameras[["camera_id", "gauge"]], on="camera_id")
    ch = ch.merge(rain, on=["gauge", "hour_ts"], how="inner")          # no rain reading -> no row
    ch = ch.dropna(subset=["rain_mm"])
    ch["wet"] = ch["rain_mm"] >= WET_MM
    ch["heavy"] = ch["rain_mm"] >= HEAVY_MM
    return add_baseline(ch)
