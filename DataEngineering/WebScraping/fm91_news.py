import argparse
import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
BASE_URL = "https://www.fm91bkk.com"
ARTICLE_URL = BASE_URL + "/newsarticle/{}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130 Safari/537.36"}
REQUEST_DELAY = 0.5
OUT_DIR = Path(__file__).resolve().parent / "raw" / "fm91_news"
OUT_FILE = OUT_DIR / "fm91_news.csv"
TH_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
             "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
DATE_RX = re.compile(r"วันที่\s*(\d{1,2})\s+(\S+)\s+(\d{4})\s+เวลา\s+(\d{1,2}):(\d{2})")
KEYWORDS = {
    "mentions_accident": r"อุบัติเหตุ|ชนกัน|พลิกคว่ำ|รถชน|เฉี่ยวชน",
    "mentions_flood": r"น้ำท่วม|น้ำขัง|ท่วมขัง|ระบายน้ำ",
    "mentions_rain": r"ฝน|พายุ",
    "mentions_congestion": r"รถติด|การจราจร(?:ติดขัด|หนาแน่น)|ติดขัด|หนาแน่น|เคลื่อนตัวได้ช้า",
}
ROAD_RX = re.compile(r"(?:ถนน|ถ\.)\s?([ก-๙A-Za-z0-9\.\- ]{2,30}?)(?=[\s,]|$)")
JUNCTION_RX = re.compile(r"(แยก[ก-๙A-Za-z0-9\.\-]{2,25})")
COLUMNS = ["article_id", "datetime", "category", "title", "body", "url", "is_traffic_category",
           "mentions_accident", "mentions_flood", "mentions_rain", "mentions_congestion",
           "bangkok_related", "roads", "intersections"]

session = requests.Session()
session.headers.update(HEADERS)


def parse_article(html: str, article_id: int):
    soup = BeautifulSoup(html, "html.parser")
    date_tag = soup.find(string=DATE_RX)
    if not date_tag:
        return None
    d, month, year, hh, mm = DATE_RX.search(date_tag).groups()
    if month not in TH_MONTHS:
        return None
    breadcrumb = next((p for p in soup.find_all("p") if p.select("a[href*='/newscategory/']")), None)
    categories = breadcrumb.select("a[href*='/newscategory/']") if breadcrumb else []
    title = breadcrumb.find_next("h1") if breadcrumb else (soup.find_all("h1") or [None])[-1]
    body = soup.select_one("div.news-pra")
    if body is None:
        meta = soup.find("meta", attrs={"property": "og:description"})
        body_text = meta.get("content", "").strip() if meta else None
    else:
        body_text = body.get_text(" ", strip=True)
    return {
        "article_id": article_id,
        "datetime": datetime(int(year) - 543, TH_MONTHS.index(month) + 1, int(d), int(hh), int(mm)),
        "category": categories[-1].get_text(strip=True) if categories else None,
        "title": title.get_text(strip=True) if title else None,
        "body": body_text or None,
        "url": ARTICLE_URL.format(article_id),
    }


def fetch_article(article_id: int):
    r = session.get(ARTICLE_URL.format(article_id), timeout=60)
    if r.status_code != 200:
        return None
    return parse_article(r.text, article_id)


def latest_article_id() -> int:
    soup = BeautifulSoup(session.get(BASE_URL + "/news", timeout=60).text, "html.parser")
    ids = [int(m.group(1)) for a in soup.select("a[href*='/newsarticle/']")
           if (m := re.search(r"/newsarticle/(\d+)", a["href"]))]
    return max(ids)


def article_date(article_id: int, cache: dict):
    for k in range(article_id, article_id + 15):
        if k not in cache:
            cache[k] = fetch_article(k)
            time.sleep(REQUEST_DELAY)
        if cache[k]:
            return cache[k]["datetime"].date()
    return None


def first_id_on_or_after(target: date, lo: int, hi: int, cache: dict) -> int:
    while lo < hi:
        mid = (lo + hi) // 2
        d = article_date(mid, cache)
        if d is None or d < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["title"].fillna("") + " " + df["body"].fillna("")
    df["is_traffic_category"] = (df["category"] == "ข่าวจราจร").astype(int)
    for col, rx in KEYWORDS.items():
        df[col] = text.str.contains(rx, regex=True).astype(int)
    df["bangkok_related"] = text.str.contains(r"กรุงเทพ|กทม\.|เขต[ก-๙]+").astype(int)
    df["roads"] = text.map(lambda s: "; ".join(dict.fromkeys(x.strip() for x in ROAD_RX.findall(s))))
    df["intersections"] = text.map(lambda s: "; ".join(dict.fromkeys(JUNCTION_RX.findall(s))))
    return df.reindex(columns=COLUMNS)


def fetch() -> pd.DataFrame:
    start, end = date.fromisoformat(START_DATE), date.fromisoformat(END_DATE)
    cache = {}
    hi = latest_article_id()
    first = first_id_on_or_after(start, 1, hi, cache)
    last = first_id_on_or_after(date.fromordinal(end.toordinal() + 1), first, hi, cache) - 1
    print(f"    article id {first} to {last} ({last - first + 1} pages)", flush=True)
    rows = []
    for i in range(first, last + 1):
        a = cache.get(i) or fetch_article(i)
        if i not in cache:
            time.sleep(REQUEST_DELAY)
        if a and start <= a["datetime"].date() <= end:
            rows.append(a)
        if (i - first) % 100 == 0:
            print(f"    read {i - first + 1}/{last - first + 1}", flush=True)
    return add_text_features(pd.DataFrame(rows))


def main(force: bool = False):
    if OUT_FILE.exists() and not force:
        print(f"[!] {OUT_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {OUT_FILE.name}: {len(df)} articles | traffic {df.is_traffic_category.sum()}"
          f" | accident {df.mentions_accident.sum()} | flood {df.mentions_flood.sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
