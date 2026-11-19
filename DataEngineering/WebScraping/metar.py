import argparse
import math
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
PAGE_URL = "https://mesonet.agron.iastate.edu/sites/obhistory.php"
NETWORK = "TH__ASOS"
HEADERS = {"User-Agent": "bkk-cctv-class-project/1.0"}
REQUEST_DELAY = 1.0
STATIONS = {
    "VTBD": ("Don Mueang", 13.9126, 100.6067),
    "VTBS": ("Suvarnabhumi", 13.6900, 100.7501),
}
OUT_DIR = Path(__file__).resolve().parent / "raw" / "metar"
OUT_FILE = OUT_DIR / "metar_hourly.csv"
COLUMNS = ["station", "station_name", "latitude", "longitude", "datetime", "temp_c", "rh", "wind_kt",
           "visibility_km", "sky_cover", "wxcodes", "metar", "is_rain", "is_heavy_rain", "is_thunderstorm",
           "is_shower"]
WIND_RX = re.compile(r"\b(?:\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b")
TEMP_RX = re.compile(r"\s(M?\d{2})/(M?\d{2})?(?=\s|$)")


def weather_flags(wxcodes: pd.Series) -> pd.DataFrame:
    wx = wxcodes.fillna("").astype(str)
    return pd.DataFrame({
        "is_rain": wx.str.contains("RA").astype(int),
        "is_heavy_rain": wx.str.contains(r"\+").astype(int),
        "is_thunderstorm": wx.str.contains("TS").astype(int),
        "is_shower": wx.str.contains("SH").astype(int),
    }, index=wxcodes.index)


def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def metar_number(text):
    if not text:
        return None
    return -float(text[1:]) if text.startswith("M") else float(text)


def temps_from_metar(metar: str):
    m = TEMP_RX.search(metar)
    return (metar_number(m.group(1)), metar_number(m.group(2))) if m else (None, None)


def saturation_vapor_pressure(t_c: float) -> float:
    t, t0, e0, l0, rv, cpl, cpv = t_c + 273.15, 273.16, 611.655, 2.501e6, 461.52, 4219.4, 1860.078
    latent = l0 - (cpl - cpv) * (t - t0)
    return e0 * (t0 / t) ** ((cpl - cpv) / rv) * math.exp(l0 / (rv * t0) - latent / (rv * t))


def relative_humidity(temp_c, dew_c):
    if temp_c is None or dew_c is None:
        return None
    return round(100 * saturation_vapor_pressure(dew_c) / saturation_vapor_pressure(temp_c), 0)


def wind_from_metar(metar: str):
    m = WIND_RX.search(metar)
    return float(m.group(1)) if m else None


def parse_day(html: str, station: str, day: str) -> list:
    soup = BeautifulSoup(html, "lxml")
    name, lat, lon = STATIONS[station]
    rows = []
    for ob in soup.select("tr.ob"):
        cells = [td.get_text(" ", strip=True) for td in ob.find_all("td")]
        if len(cells) < 11:
            continue
        nxt = ob.find_next_sibling("tr")
        metar = nxt.get_text(" ", strip=True) if nxt is not None and "metar" in (nxt.get("class") or []) else ""
        vis_mi = to_float(cells[2])
        temp, dew = temps_from_metar(metar)
        if temp is None and to_float(cells[5]) is not None:
            temp = (to_float(cells[5]) - 32) * 5 / 9
        rh = relative_humidity(temp, dew)
        rows.append({
            "station": station,
            "station_name": name,
            "latitude": lat,
            "longitude": lon,
            "datetime": datetime.strptime(f"{day} {cells[0]}", "%Y-%m-%d %I:%M %p"),
            "temp_c": round(temp, 1) if temp is not None else None,
            "rh": rh if rh is not None else to_float(cells[10]),
            "wind_kt": wind_from_metar(metar),
            "visibility_km": round(vis_mi * 1.609, 1) if vis_mi is not None else None,
            "sky_cover": cells[3].split()[0][:3] if cells[3] else None,
            "wxcodes": cells[4],
            "metar": metar or None,
        })
    return rows


def fetch() -> pd.DataFrame:
    rows = []
    for station in STATIONS:
        for day in pd.date_range(START_DATE, END_DATE).strftime("%Y-%m-%d"):
            r = requests.get(PAGE_URL, headers=HEADERS, timeout=60,
                             params={"station": station, "network": NETWORK, "date": day})
            r.raise_for_status()
            day_rows = parse_day(r.text, station, day)
            rows += day_rows
            print(f"    {station} {day}: {len(day_rows)} reports", flush=True)
            time.sleep(REQUEST_DELAY)
    df = pd.DataFrame(rows).sort_values(["datetime", "station"], ignore_index=True)
    df = pd.concat([df, weather_flags(df["wxcodes"])], axis=1)
    return df.reindex(columns=COLUMNS)


def main(force: bool = False):
    if OUT_FILE.exists() and not force:
        print(f"[!] {OUT_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {OUT_FILE.name}: {len(df):,} reports | rain {df.is_rain.sum()} | thunderstorm {df.is_thunderstorm.sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
