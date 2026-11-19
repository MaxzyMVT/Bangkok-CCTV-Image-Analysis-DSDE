import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

START_DATE = "2026-08-25"
END_DATE = "2026-08-31"
PAGE_URL = "https://www.bot.or.th/th/financial-institutions-holiday.html"
SOURCE = "bot.or.th"
WAIT_SECONDS = 60
OUT_DIR = Path(__file__).resolve().parent / "raw" / "bot_holidays"
HOLIDAYS_FILE = OUT_DIR / "thai_holidays.csv"
CALENDAR_FILE = OUT_DIR / "calendar_features.csv"
TH_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
             "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
CALENDAR_COLUMNS = ["date", "day_of_week", "is_weekend", "holiday_name", "is_public_holiday", "is_day_off",
                    "is_long_weekend", "is_workday_before_holiday", "is_workday_after_holiday"]


def open_browser():
    from selenium import webdriver

    try:
        options = webdriver.EdgeOptions()
        options.add_argument("--headless=new")
        return webdriver.Edge(options=options)
    except Exception:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        return webdriver.Chrome(options=options)


def shown_year(driver) -> int:
    from selenium.webdriver.common.by import By

    return int(driver.find_element(By.CSS_SELECTOR, ".holiday-inner-section .section-title h3").text.strip())


def show_year(driver, be_year: int):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    for _ in range(15):
        current = shown_year(driver)
        if current == be_year:
            return
        arrow = ".left" if current > be_year else ".right"
        button = driver.find_element(By.CSS_SELECTOR, f".holiday-inner-section .arrow-next {arrow}")
        driver.execute_script("arguments[0].click();", button)
        WebDriverWait(driver, WAIT_SECONDS).until(lambda d: shown_year(d) != current)
    raise RuntimeError(f"year {be_year} is not available on {PAGE_URL}")


def parse_holidays(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    section = soup.select_one(".holiday-inner-section")
    be_year = int(section.select_one(".section-title h3").get_text(strip=True))
    rows = []
    for group in section.select(".holiday-group"):
        month = group.select_one(".month-title h3").get_text(strip=True)
        for item in group.select(".month-holiday-item .item-desc"):
            day_text, name = (h.get_text(" ", strip=True) for h in item.find_all("h3")[:2])
            day = int(re.search(r"\d+", day_text).group())
            rows.append({"date": date(be_year - 543, TH_MONTHS.index(month) + 1, day),
                         "holiday_name": name, "source": SOURCE})
    return pd.DataFrame(rows, columns=["date", "holiday_name", "source"])


def fetch_holidays(years) -> pd.DataFrame:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    driver = open_browser()
    try:
        driver.get(PAGE_URL)
        WebDriverWait(driver, WAIT_SECONDS).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, ".holiday-inner-section .month-holiday-item"))
        frames = []
        for year in years:
            show_year(driver, year + 543)
            frames.append(parse_holidays(driver.page_source))
            print(f"    {year}: {len(frames[-1])} holidays", flush=True)
    finally:
        driver.quit()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "holiday_name"]).sort_values("date")


def build_calendar(holidays: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=7), pd.Timestamp(end) + pd.Timedelta(days=7))
    cal = pd.DataFrame({"date": days})
    names = holidays.assign(date=pd.to_datetime(holidays["date"])).groupby("date")["holiday_name"].agg("; ".join)
    cal["day_of_week"] = cal["date"].dt.day_name()
    cal["is_weekend"] = (cal["date"].dt.day_of_week >= 5).astype(int)
    cal["holiday_name"] = cal["date"].map(names)
    cal["is_public_holiday"] = cal["holiday_name"].notna().astype(int)
    cal["is_day_off"] = ((cal["is_weekend"] == 1) | (cal["is_public_holiday"] == 1)).astype(int)
    run_id = (cal["is_day_off"] != cal["is_day_off"].shift()).cumsum()
    run_length = cal.groupby(run_id)["is_day_off"].transform("size")
    cal["is_long_weekend"] = ((cal["is_day_off"] == 1) & (run_length >= 3)).astype(int)
    cal["is_workday_before_holiday"] = ((cal["is_day_off"] == 0) & (cal["is_day_off"].shift(-1) == 1)).astype(int)
    cal["is_workday_after_holiday"] = ((cal["is_day_off"] == 0) & (cal["is_day_off"].shift(1) == 1)).astype(int)
    cal = cal[cal["date"].between(start, end)].copy()
    cal["date"] = cal["date"].dt.date
    return cal[CALENDAR_COLUMNS].reset_index(drop=True)


def main(force: bool = False):
    if HOLIDAYS_FILE.exists() and CALENDAR_FILE.exists() and not force:
        print(f"[!] {HOLIDAYS_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first = pd.Timestamp(START_DATE) - pd.Timedelta(days=7)
    last = pd.Timestamp(END_DATE) + pd.Timedelta(days=7)
    holidays = fetch_holidays(range(first.year, last.year + 1))
    holidays.to_csv(HOLIDAYS_FILE, index=False, encoding="utf-8-sig")
    calendar = build_calendar(holidays, START_DATE, END_DATE)
    calendar.to_csv(CALENDAR_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {HOLIDAYS_FILE.name}: {len(holidays)} holidays | {CALENDAR_FILE.name}: {len(calendar)} days,"
          f" {calendar.is_day_off.sum()} days off")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
