import argparse
import io
import json
from pathlib import Path

import pandas as pd
import requests

CKAN_URL = "https://data.bangkok.go.th/api/3/action/package_show"
GIS_URL = ("https://cpudgiportal.bangkok.go.th/arcgis/rest/services/Enterprise_Site_Service/"
           "SUBDISTRICT_BANGKOK_2552/MapServer/1/query")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130 Safari/537.36"}
POPULATION_DATASET = "bkkpopulationdistric"
HAWKER_DATASET = "ced37"
OUT_DIR = Path(__file__).resolve().parent / "raw" / "bma_open_data"
POPULATION_FILE = OUT_DIR / "bma_population_district.csv"
STATS_FILE = OUT_DIR / "bma_district_stats.csv"
DISTRICTS_FILE = OUT_DIR / "bma_districts.geojson"
HAWKER_FILE = OUT_DIR / "bma_hawker_zones.csv"


def download_csv(package_id: str) -> pd.DataFrame:
    meta = requests.get(CKAN_URL, params={"id": package_id}, headers=HEADERS, timeout=60).json()["result"]
    res = next(r for r in meta["resources"] if (r.get("format") or "").upper() == "CSV")
    raw = requests.get(res["url"], headers=HEADERS, timeout=120)
    raw.raise_for_status()
    for enc in ("utf-8-sig", "cp874"):
        try:
            return pd.read_csv(io.BytesIO(raw.content), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {package_id}")


def fetch_population() -> pd.DataFrame:
    pop = download_csv(POPULATION_DATASET)
    pop.columns = [c.strip().lower() for c in pop.columns]
    pop = pop.rename(columns={"district": "district_name", "total": "population"})
    pop["district_name"] = pop["district_name"].astype(str).str.strip()
    return pop[["no", "district_name", "population", "male", "female"]]


def fetch_districts() -> dict:
    r = requests.get(GIS_URL, headers=HEADERS, timeout=120,
                     params={"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson"})
    r.raise_for_status()
    return r.json()


def build_stats(geojson: dict, pop: pd.DataFrame) -> pd.DataFrame:
    dist = pd.DataFrame([f["properties"] for f in geojson["features"]])
    dist["district_name"] = dist["DISTRICT_NAME_T"].str.replace("^เขต", "", regex=True).str.strip()
    dist["area_km2"] = (dist["SHAPE.AREA"] / 1e6).round(2)
    dist = dist.merge(pop[["district_name", "population"]], on="district_name", how="left")
    dist["pop_density_per_km2"] = (dist["population"] / dist["area_km2"]).round(0)
    return dist[["DISTRICT_CODE", "district_name", "DISTRICT_NAME_E", "area_km2", "population",
                 "pop_density_per_km2"]]


def main(force: bool = False):
    files = (POPULATION_FILE, STATS_FILE, DISTRICTS_FILE, HAWKER_FILE)
    if all(f.exists() for f in files) and not force:
        print(f"[!] {POPULATION_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pop = fetch_population()
    pop.to_csv(POPULATION_FILE, index=False, encoding="utf-8-sig")
    geojson = fetch_districts()
    DISTRICTS_FILE.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    stats = build_stats(geojson, pop)
    stats.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    hawker = download_csv(HAWKER_DATASET).rename(columns=str.strip)
    hawker.to_csv(HAWKER_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {POPULATION_FILE.name}: {len(pop)} rows | {DISTRICTS_FILE.name}: {len(geojson['features'])} districts"
          f" | {HAWKER_FILE.name}: {len(hawker)} zones")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
