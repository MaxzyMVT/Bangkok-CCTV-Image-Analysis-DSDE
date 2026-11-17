import argparse
import io
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
API_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
HEADERS = {"User-Agent": "bkk-cctv-class-project/1.0"}
STATIONS = {
    "VTBD": ("Don Mueang", 13.9126, 100.6067),
    "VTBS": ("Suvarnabhumi", 13.6900, 100.7501),
}
OUT_DIR = Path(__file__).resolve().parent / "raw" / "metar"
OUT_FILE = OUT_DIR / "metar_hourly.csv"


def weather_flags(wxcodes: pd.Series) -> pd.DataFrame:
    wx = wxcodes.fillna("").astype(str)
    return pd.DataFrame({
        "is_rain": wx.str.contains("RA").astype(int),
        "is_heavy_rain": wx.str.contains(r"\+").astype(int),
        "is_thunderstorm": wx.str.contains("TS").astype(int),
        "is_shower": wx.str.contains("SH").astype(int),
    }, index=wxcodes.index)


def fetch() -> pd.DataFrame:
    s = date.fromisoformat(START_DATE)
    e = date.fromisoformat(END_DATE) + timedelta(days=1)
    params = [("station", k) for k in STATIONS] + [
        ("data", "all"), ("tz", "Asia/Bangkok"), ("format", "onlycomma"),
        ("missing", "empty"), ("trace", "T"),
        ("year1", s.year), ("month1", s.month), ("day1", s.day),
        ("year2", e.year), ("month2", e.month), ("day2", e.day),
    ]
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def build(m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "station": m["station"],
        "station_name": m["station"].map(lambda k: STATIONS[k][0]),
        "latitude": m["station"].map(lambda k: STATIONS[k][1]),
        "longitude": m["station"].map(lambda k: STATIONS[k][2]),
        "datetime": pd.to_datetime(m["valid"]),
        "temp_c": ((m["tmpf"] - 32) * 5 / 9).round(1),
        "rh": m["relh"].round(0),
        "wind_kt": m["sknt"],
        "visibility_km": (m["vsby"] * 1.609).round(1),
        "sky_cover": m["skyc1"],
        "wxcodes": m["wxcodes"].fillna(""),
        "metar": m["metar"],
    })
    out = pd.concat([out, weather_flags(out["wxcodes"])], axis=1)
    keep = out["datetime"].dt.date.between(date.fromisoformat(START_DATE), date.fromisoformat(END_DATE))
    return out[keep]


def main(force: bool = False):
    if OUT_FILE.exists() and not force:
        print(f"[!] {OUT_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build(fetch())
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {OUT_FILE.name}: {len(df):,} reports | rain {df.is_rain.sum()} | thunderstorm {df.is_thunderstorm.sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
