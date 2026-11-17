import argparse
from pathlib import Path

import pandas as pd
import requests

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
LATITUDE = 13.7563
LONGITUDE = 100.5018
API_URL = "https://archive-api.open-meteo.com/v1/archive"
HEADERS = {"User-Agent": "bkk-cctv-class-project/1.0"}
OUT_DIR = Path(__file__).resolve().parent / "raw" / "sun_times"
OUT_FILE = OUT_DIR / "sun_times_daily.csv"


def fetch() -> pd.DataFrame:
    r = requests.get(API_URL, headers=HEADERS, timeout=60, params={
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": "sunrise,sunset,daylight_duration", "timezone": "Asia/Bangkok",
    })
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    d["sunrise"] = pd.to_datetime(d["sunrise"])
    d["sunset"] = pd.to_datetime(d["sunset"])
    d["daylight_hours"] = (d.pop("daylight_duration") / 3600).round(2)
    return d[["date", "sunrise", "sunset", "daylight_hours"]]


def main(force: bool = False):
    if OUT_FILE.exists() and not force:
        print(f"[!] {OUT_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {OUT_FILE.name}: {len(df)} days")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
