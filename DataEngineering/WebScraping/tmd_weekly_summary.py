import argparse
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
BASE_URL = "https://www.tmd.go.th"
LIST_URL = BASE_URL + "/climate/summaryweekly"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130 Safari/537.36"}
REQUEST_DELAY = 1.0
OUT_DIR = Path(__file__).resolve().parent / "raw" / "tmd_weekly_summary"
OUT_FILE = OUT_DIR / "tmd_weekly_summary.csv"
TH_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
             "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
TH_MONTHS_SHORT = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
                   "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
RANGE_RX = re.compile(r"(\d{1,2})\s+(\S+)\s+(\d{4})\s*-\s*(\d{1,2})\s+(\S+)\s+(\d{4})")
VALUE_RX = re.compile(r"(\d+(?:\.\d+)?)\s*มม\.\s*ที่\s*(.*?)\s*เมื่อวันที่\s*(\d{1,2})\s*(\S+\.\S+\.)", re.S)
PERCENT_RX = re.compile(r"ร้อยละ\s*(\d{1,3})(?:\s*-\s*(\d{1,3}))?")
COLUMNS = ["week_start", "week_end", "region", "summary", "rain_area_pct_min", "rain_area_pct_max",
           "mentions_heavy_rain", "mentions_very_heavy_rain", "mentions_flood", "mentions_flash_flood",
           "max_rain_mm", "max_rain_location", "max_rain_date", "url"]


def thai_date(day: str, month: str, be_year: str) -> date:
    return date(int(be_year) - 543, TH_MONTHS.index(month) + 1, int(day))


def parse_week_list(html: str) -> list:
    soup = BeautifulSoup(html, "lxml")
    weeks = []
    for option in soup.select("select option"):
        m = RANGE_RX.search(option.get_text(" ", strip=True))
        value = option.get("value") or ""
        if not m or "/climate/summaryweekly/" not in value:
            continue
        d1, m1, y1, d2, m2, y2 = m.groups()
        if m1 in TH_MONTHS and m2 in TH_MONTHS:
            weeks.append({"week_start": thai_date(d1, m1, y1), "week_end": thai_date(d2, m2, y2),
                          "url": BASE_URL + value})
    return weeks


def parse_value(text: str, week_start: date, week_end: date):
    m = VALUE_RX.search(text or "")
    if not m or m.group(4) not in TH_MONTHS_SHORT:
        return None, None, None
    month = TH_MONTHS_SHORT.index(m.group(4)) + 1
    year = week_start.year if month == week_start.month else week_end.year
    return float(m.group(1)), " ".join(m.group(2).split()), date(year, month, int(m.group(3)))


def parse_week_page(html: str, week: dict) -> list:
    soup = BeautifulSoup(html, "lxml")
    heading = next((h for h in soup.find_all("h3") if "รายภาค" in h.get_text()), None)
    container = heading.find_parent(class_="text-start") if heading else soup
    rows = []
    for title in container.select("p.sub-title"):
        block = title.parent
        summary_tag = block.select_one("p.sub-content")
        summary = " ".join(summary_tag.get_text(" ", strip=True).split()) if summary_tag else ""
        cells = [td.get_text(" ", strip=True) for td in block.select("table tbody tr td")] or [""]
        percents = [int(x) for pair in PERCENT_RX.findall(summary) for x in pair if x]
        max_rain = parse_value(cells[0], week["week_start"], week["week_end"])
        rows.append({
            "week_start": week["week_start"], "week_end": week["week_end"],
            "region": title.get_text(" ", strip=True), "summary": summary,
            "rain_area_pct_min": min(percents) if percents else None,
            "rain_area_pct_max": max(percents) if percents else None,
            "mentions_heavy_rain": int("ฝนหนัก" in summary),
            "mentions_very_heavy_rain": int("หนักมาก" in summary),
            "mentions_flood": int("น้ำท่วม" in summary),
            "mentions_flash_flood": int("น้ำป่า" in summary),
            "max_rain_mm": max_rain[0], "max_rain_location": max_rain[1], "max_rain_date": max_rain[2],
            "url": week["url"],
        })
    return rows


def fetch() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(HEADERS)
    weeks = parse_week_list(session.get(LIST_URL, timeout=60).text)
    start, end = date.fromisoformat(START_DATE), date.fromisoformat(END_DATE)
    selected = [w for w in weeks if w["week_start"] <= end and w["week_end"] >= start]
    rows = []
    for week in selected:
        time.sleep(REQUEST_DELAY)
        r = session.get(week["url"], timeout=60)
        r.raise_for_status()
        week_rows = parse_week_page(r.text, week)
        rows += week_rows
        print(f"    {week['week_start']} to {week['week_end']}: {len(week_rows)} regions", flush=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["week_start", "region"], ignore_index=True)


def main(force: bool = False):
    if OUT_FILE.exists() and not force:
        print(f"[!] {OUT_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {OUT_FILE.name}: {len(df)} rows ({df.week_start.nunique()} weeks x {df.region.nunique()} regions)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
