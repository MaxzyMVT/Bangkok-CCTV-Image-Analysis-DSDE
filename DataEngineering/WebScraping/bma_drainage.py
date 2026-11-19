import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
BASE_URL = "https://weather.bangkok.go.th"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130 Safari/537.36"}
CANAL_STEP_HOURS = 12
REQUEST_DELAY = 0.5
OUT_DIR = Path(__file__).resolve().parent / "raw" / "bma_drainage"
RAIN_FILE = OUT_DIR / "dds_rain_hourly.csv"
FLOOD_FILE = OUT_DIR / "dds_flood_hourly.csv"
CANAL_FILE = OUT_DIR / "dds_canal_level.csv"
STATIONS_FILE = OUT_DIR / "dds_stations.csv"
SYSTEMS = ("rain", "flood", "canal")

READ_TABLE_JS = r"""(id) => {
  if (window.jQuery && jQuery.fn.dataTable && jQuery.fn.dataTable.isDataTable('#' + id)) {
    jQuery('#' + id).DataTable().page.len(-1).draw(false);
  }
  const t = document.getElementById(id);
  if (!t) return [];
  return [...t.tBodies[0].rows].map(r => [...r.cells].map(c => c.innerText.trim()));
}"""


def hours() -> pd.DatetimeIndex:
    return pd.date_range(f"{START_DATE} 00:00", f"{END_DATE} 23:00", freq="h")


def form_date(ts: pd.Timestamp):
    return ts.strftime("%d/%m/%Y"), ts.strftime("%H:%M")


def to_number(text):
    return pd.to_numeric(re.sub(r"[^\d\.\-]", "", text or ""), errors="coerce")


def submit(page, js: str):
    with page.expect_navigation(timeout=120_000):
        page.evaluate(js)
    page.wait_for_load_state("networkidle", timeout=120_000)


def scrape_rain(page) -> pd.DataFrame:
    page.goto(f"{BASE_URL}/rain/RainHistory/IndexAllStation", wait_until="networkidle", timeout=120_000)
    page.click("a[href='#tab-2']")
    rows = []
    for i, h in enumerate(hours()):
        d1, t1 = form_date(h)
        d2, t2 = form_date(h + pd.Timedelta(hours=1))
        submit(page, f"""() => {{
            document.cookie = 'tab=2';
            const f = document.querySelector('#FrmIndextab2');
            f.querySelector('[name=datePick]').value = '{d1}';
            f.querySelector('[name=StationTime]').value = '{t1}';
            f.querySelector('[name=datePick_Stop]').value = '{d2}';
            f.querySelector('[name=StationTime_Stop]').value = '{t2}';
            f.submit(); }}""")
        for r in page.evaluate(READ_TABLE_JS, "example1"):
            if len(r) >= 5:
                rows.append({"station_code": r[0], "district": r[1], "station": r[2],
                             "datetime": h, "rain_mm": to_number(r[-1])})
        if i % 24 == 0:
            print(f"    rain {h:%Y-%m-%d} ({len(rows):,} rows)", flush=True)
        time.sleep(REQUEST_DELAY)
    return pd.DataFrame(rows)


def scrape_flood(page) -> pd.DataFrame:
    page.goto(f"{BASE_URL}/Flood/Graph/AllStation", wait_until="networkidle", timeout=120_000)
    rows = []
    for i, h in enumerate(hours()):
        d, t = form_date(h)
        submit(page, f"""() => {{ const f = document.forms[0];
            f.datePick.value = '{d}'; f.StationTime.value = '{t}'; f.submit(); }}""")
        for r in page.evaluate(READ_TABLE_JS, "example"):
            if len(r) >= 5:
                rows.append({"station_code": r[0], "road": r[1], "station": r[2],
                             "datetime": h, "flood_cm": to_number(r[4])})
        if i % 24 == 0:
            print(f"    flood {h:%Y-%m-%d} ({len(rows):,} rows)", flush=True)
        time.sleep(REQUEST_DELAY)
    return pd.DataFrame(rows)


def scrape_canal(page) -> pd.DataFrame:
    page.goto(f"{BASE_URL}/water/WaterHistory/IndexStations", wait_until="networkidle", timeout=120_000)
    canals = page.evaluate("""() => [...document.querySelector('[name=water_station_allstation]').options]
                                     .map(o => o.value).filter(v => v)""")
    slots = hours()[::CANAL_STEP_HOURS]
    print(f"    canal: {len(canals)} canals x {len(slots)} time slots", flush=True)
    rows = []
    for h in slots:
        d, t = form_date(h)
        for canal_id in canals:
            submit(page, f"""() => {{ const f = document.forms[0];
                f.water_station_allstation.value = '{canal_id}'; f.datePick_start.value = '{d}';
                f.StationTime_start.value = '{t}'; f.submit(); }}""")
            for r in page.evaluate(READ_TABLE_JS, "example"):
                if len(r) >= 7:
                    rows.append({"station_code": r[0], "canal": r[1], "station": r[2], "datetime": h,
                                 "level_inside_m": to_number(r[4]), "level_outside_m": to_number(r[5]),
                                 "level_river_m": to_number(r[6])})
            time.sleep(REQUEST_DELAY / 2)
        print(f"    canal {h:%Y-%m-%d %H:%M} ({len(rows):,} rows)", flush=True)
    return pd.DataFrame(rows)


def fetch_stations() -> pd.DataFrame:
    s = requests.Session()
    s.headers.update(HEADERS)
    rain = pd.DataFrame(s.post(f"{BASE_URL}/rain/PageMap/GetDataForUpdate", timeout=60).json())
    water = pd.DataFrame(s.post(f"{BASE_URL}/water/PageMap/GoogleMap",
                                data={"payload": "TEST_DATA_GOES_HERE"}, timeout=60).json())
    parts = [
        rain.assign(system="rain").rename(columns={"rain_code": "station_code", "rain_shortname": "station"}),
        water.assign(system="water").rename(columns={"water_code": "station_code", "water_shortname": "station"}),
    ]
    return pd.concat([p[["system", "station_code", "station", "district_name", "latitude", "longitude"]]
                      for p in parts], ignore_index=True).sort_values(["system", "station_code"], ignore_index=True)


def launch_browser(p):
    try:
        return p.chromium.launch(channel="msedge", headless=True)
    except Exception:
        return p.chromium.launch(headless=True)


def main(force: bool = False, only=SYSTEMS):
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if force or not STATIONS_FILE.exists():
        st = fetch_stations()
        st.to_csv(STATIONS_FILE, index=False, encoding="utf-8-sig")
        print(f"[✓] {STATIONS_FILE.name}: {len(st)} stations")
    jobs = [(name, fn, path) for name, fn, path in
            (("rain", scrape_rain, RAIN_FILE), ("flood", scrape_flood, FLOOD_FILE), ("canal", scrape_canal, CANAL_FILE))
            if name in only and (force or not path.exists())]
    if not jobs:
        print("[!] drainage files exist (skipped, use --force to re-scrape)")
        return
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(locale="th-TH")
        for name, fn, path in jobs:
            df = fn(page)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[✓] {path.name}: {len(df):,} rows")
        browser.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", default=",".join(SYSTEMS), help="comma list of: rain,flood,canal")
    args = p.parse_args()
    main(args.force, tuple(s.strip() for s in args.only.split(",")))
