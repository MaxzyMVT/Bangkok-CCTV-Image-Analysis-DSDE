"""BTS/MRT stations to look at first for covered walkways: near cameras that get denser in rain."""
import pandas as pd

from DataAnalysis.rain.common import haversine_m

RADIUS_M = 500


def policy_stations(cam_effect: pd.DataFrame, ch: pd.DataFrame, cameras: pd.DataFrame, rail: pd.DataFrame,
                    radius_m: float = RADIUS_M) -> pd.DataFrame:
    """Stations within radius_m of at least one "denser" camera, ranked by how many motorcycles those cameras
    lose per wet hour (dry baseline minus wet count, averaged over their wet hours)."""
    denser = cam_effect.loc[cam_effect["label"] == "denser", "camera_id"]
    pos = cameras.set_index("camera_id")
    wet = ch[ch["wet"] & ch["camera_id"].isin(denser)]
    lost = (wet["motorcycle_base"] - wet["motorcycle"]).groupby(wet["camera_id"]).agg(["sum", "size"])
    rows = []
    for s in rail.itertuples(index=False):
        cams = [c for c in denser
                if haversine_m(s.latitude, s.longitude, pos.at[c, "latitude"], pos.at[c, "longitude"]) <= radius_m]
        cams = [c for c in cams if c in lost.index]
        if not cams:
            continue
        rows.append({"station": s.name, "latitude": s.latitude, "longitude": s.longitude, "denser_cameras": len(cams),
                     "moto_lost_per_wet_hour": float(lost.loc[cams, "sum"].sum() / lost.loc[cams, "size"].sum()),
                     "camera_ids": " ".join(str(c) for c in cams)})
    cols = ["station", "latitude", "longitude", "denser_cameras", "moto_lost_per_wet_hour", "camera_ids"]
    return (pd.DataFrame(rows, columns=cols).sort_values("moto_lost_per_wet_hour", ascending=False)
              .reset_index(drop=True))
