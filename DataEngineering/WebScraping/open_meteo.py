import argparse
import time
from pathlib import Path

import pandas as pd
import requests

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
API_URL = "https://archive-api.open-meteo.com/v1/archive"
HEADERS = {"User-Agent": "bkk-cctv-class-project/1.0"}
VARIABLES = [
    "precipitation", "rain", "temperature_2m", "relative_humidity_2m",
    "cloud_cover", "weather_code", "wind_speed_10m", "is_day",
]
POINTS = {
    "center": (13.7563, 100.5018), "north": (13.88, 100.57), "northeast": (13.85, 100.70),
    "east": (13.75, 100.75), "southeast": (13.66, 100.65), "south": (13.66, 100.50),
    "southwest": (13.68, 100.40), "west": (13.77, 100.40), "northwest": (13.84, 100.47),
}
OUT_DIR = Path(__file__).resolve().parent / "raw" / "open_meteo"
POINTS_FILE = OUT_DIR / "weather_hourly_points.csv"
CITY_FILE = OUT_DIR / "weather_hourly_bangkok.csv"


def fetch() -> list:
    params = {
        "latitude": ",".join(str(v[0]) for v in POINTS.values()),
        "longitude": ",".join(str(v[1]) for v in POINTS.values()),
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": ",".join(VARIABLES), "timezone": "Asia/Bangkok",
    }
    for attempt in range(4):
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else [data]
    raise RuntimeError("Open-Meteo rate limit reached, try again later")


def build_points(results: list) -> pd.DataFrame:
    frames = []
    for (name, (lat, lon)), res in zip(POINTS.items(), results):
        df = pd.DataFrame(res["hourly"])
        df.insert(0, "point", name)
        df.insert(1, "latitude", lat)
        df.insert(2, "longitude", lon)
        frames.append(df)
    w = pd.concat(frames, ignore_index=True).rename(columns={"time": "datetime"})
    w["datetime"] = pd.to_datetime(w["datetime"])
    w["is_raining"] = (w["precipitation"] >= 0.1).astype(int)
    w["rain_level"] = pd.cut(w["precipitation"], bins=[-0.01, 0.09, 2.5, 7.6, 1e9],
                             labels=["no_rain", "light", "moderate", "heavy"])
    return w


def build_city(w: pd.DataFrame) -> pd.DataFrame:
    return (w.groupby("datetime")
             .agg(precipitation_mean=("precipitation", "mean"),
                  precipitation_max=("precipitation", "max"),
                  pct_points_raining=("is_raining", "mean"),
                  temperature_2m=("temperature_2m", "mean"),
                  relative_humidity_2m=("relative_humidity_2m", "mean"),
                  cloud_cover=("cloud_cover", "mean"))
             .reset_index())


def main(force: bool = False):
    if POINTS_FILE.exists() and CITY_FILE.exists() and not force:
        print(f"[!] {POINTS_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = build_points(fetch())
    city = build_city(points)
    points.to_csv(POINTS_FILE, index=False, encoding="utf-8-sig")
    city.to_csv(CITY_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {POINTS_FILE.name}: {len(points):,} rows | {CITY_FILE.name}: {len(city)} hours")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
