import argparse
import time
from pathlib import Path

import pandas as pd
import requests

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
API_URL = "https://publicapi.traffy.in.th/share/teamchadchart/search"
HEADERS = {"User-Agent": "bkk-cctv-class-project/1.0"}
PAGE_SIZE = 1000
OUT_DIR = Path(__file__).resolve().parent / "raw" / "traffy_fondue"
OUT_FILE = OUT_DIR / "traffy_fondue.csv"
COLUMNS = ["ticket_id", "timestamp", "type", "description", "address", "district", "org", "state", "star",
           "count_reopen", "latitude", "longitude", "photo_url", "after_photo", "note", "last_activity",
           "resolve_hours", "is_flood", "is_road", "is_traffic", "is_sidewalk"]


def get_page(day: str, offset: int) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(API_URL, headers=HEADERS, timeout=120, params={
                "start": day, "end": day, "limit": PAGE_SIZE, "offset": offset, "sort": "asc"})
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Traffy API did not respond")


def fetch() -> list:
    days = pd.date_range(pd.Timestamp(START_DATE) - pd.Timedelta(days=1), END_DATE).strftime("%Y-%m-%d")
    rows = []
    for day in days:
        offset, total = 0, None
        while total is None or offset < total:
            page = get_page(day, offset)
            total = page["total"]
            rows += page["results"]
            offset += PAGE_SIZE
            time.sleep(1)
        print(f"    {day} (UTC): {total:,} tickets", flush=True)
    return rows


def build(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows).drop_duplicates("ticket_id")
    df["longitude"] = pd.to_numeric(df["coords"].str[0], errors="coerce")
    df["latitude"] = pd.to_numeric(df["coords"].str[1], errors="coerce")
    for c in ("timestamp", "last_activity"):
        df[c] = pd.to_datetime(df[c], utc=True, format="mixed").dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
    df = df[df["timestamp"].dt.date.between(pd.Timestamp(START_DATE).date(), pd.Timestamp(END_DATE).date())].copy()
    df["type"] = df["problem_type_abdul"].map(
        lambda x: ",".join(t for t in (x or []) if t) if isinstance(x, list) else x)
    df["district"] = df["address"].str.extract(r"เขต\s?([ก-๙]+)")
    df["resolve_hours"] = ((df["last_activity"] - df["timestamp"]).dt.total_seconds() / 3600).where(
        df["state"] == "เสร็จสิ้น").round(1)
    text = df["type"].fillna("") + " " + df["description"].fillna("")
    df["is_flood"] = text.str.contains("น้ำท่วม|ท่วม|น้ำขัง|ระบายน้ำ|ท่อ").astype(int)
    df["is_road"] = text.str.contains("ถนน|หลุม|ผิวทาง").astype(int)
    df["is_traffic"] = text.str.contains("จราจร|รถติด|จอดรถ|กีดขวาง").astype(int)
    df["is_sidewalk"] = text.str.contains("ทางเท้า|หาบเร่|แผงลอย").astype(int)
    return df.reindex(columns=COLUMNS)


def main(force: bool = False):
    if OUT_FILE.exists() and not force:
        print(f"[!] {OUT_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build(fetch())
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {OUT_FILE.name}: {len(df):,} tickets (Bangkok time {START_DATE} to {END_DATE})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
