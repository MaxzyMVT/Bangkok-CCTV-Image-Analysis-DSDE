import argparse
import time
from pathlib import Path

import pandas as pd
import requests

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
HEADERS = {"User-Agent": "bkk-cctv-class-project/1.0"}
BBOX = "(13.45,100.30,14.10,100.95)"
POI_QUERIES = {
    "school": 'nwr["amenity"~"^(school|college|university|kindergarten)$"]',
    "hospital": 'nwr["amenity"~"^(hospital|clinic)$"]',
    "mall": 'nwr["shop"~"^(mall|department_store)$"]',
    "market": 'nwr["amenity"="marketplace"]',
    "bus_stop": 'node["highway"="bus_stop"]',
    "rail_station": 'nwr["railway"="station"]["station"~"subway|light_rail|monorail"]',
    "rail_station_all": 'nwr["public_transport"="station"]["railway"="station"]',
    "footbridge": 'way["highway"="footway"]["bridge"="yes"]',
    "traffic_signals": 'node["highway"="traffic_signals"]',
}
ROAD_TYPES = "^(motorway|trunk|primary|secondary|tertiary)$"
OUT_DIR = Path(__file__).resolve().parent / "raw" / "openstreetmap"
POI_FILE = OUT_DIR / "osm_poi.csv"
ROADS_FILE = OUT_DIR / "osm_roads.csv"


def overpass(query: str, timeout: int = 180) -> dict:
    for attempt in range(6):
        url = MIRRORS[attempt % len(MIRRORS)]
        try:
            r = requests.post(url, data={"data": f"[out:json][timeout:{timeout}];{query}"},
                              headers=HEADERS, timeout=timeout + 30)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("Overpass API is busy, try again later")


def center(el: dict):
    return (el.get("lat") or el.get("center", {}).get("lat"),
            el.get("lon") or el.get("center", {}).get("lon"))


def fetch_pois() -> pd.DataFrame:
    rows = []
    for cat, selector in POI_QUERIES.items():
        data = overpass(f"{selector}{BBOX};out center tags;")
        for el in data["elements"]:
            lat, lon = center(el)
            t = el.get("tags", {})
            rows.append({"category": cat, "osm_type": el["type"], "osm_id": el["id"],
                         "name": t.get("name"), "name_en": t.get("name:en"), "network": t.get("network"),
                         "latitude": lat, "longitude": lon})
        print(f"    {cat}: {len(data['elements']):,}", flush=True)
        time.sleep(2)
    df = pd.DataFrame(rows)
    stations = df[df.category.isin(["rail_station", "rail_station_all"])].drop_duplicates(["osm_type", "osm_id"])
    return pd.concat([df[~df.category.isin(["rail_station", "rail_station_all"])],
                      stations.assign(category="rail_station")], ignore_index=True)


def fetch_roads() -> pd.DataFrame:
    data = overpass(f'way["highway"~"{ROAD_TYPES}"]{BBOX};out center tags;', timeout=300)
    rows = []
    for el in data["elements"]:
        t = el.get("tags", {})
        lat, lon = center(el)
        rows.append({"osm_id": el["id"], "road_name": t.get("name"), "road_name_en": t.get("name:en"),
                     "highway": t.get("highway"), "lanes": pd.to_numeric(t.get("lanes"), errors="coerce"),
                     "oneway": t.get("oneway"), "maxspeed": t.get("maxspeed"), "ref": t.get("ref"),
                     "latitude": lat, "longitude": lon})
    return pd.DataFrame(rows)


def main(force: bool = False):
    if POI_FILE.exists() and ROADS_FILE.exists() and not force:
        print(f"[!] {POI_FILE.name} exists (skipped, use --force to re-scrape)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    poi = fetch_pois()
    poi.to_csv(POI_FILE, index=False, encoding="utf-8-sig")
    roads = fetch_roads()
    roads.to_csv(ROADS_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] {POI_FILE.name}: {len(poi):,} places | {ROADS_FILE.name}: {len(roads):,} road segments")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
