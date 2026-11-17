import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WEB_DIR = Path(__file__).resolve().parent
RAW_DIR = WEB_DIR / "raw"
SCRAPERS = [
    "sun_times.py", "open_meteo.py", "metar.py", "osm_overpass.py", "bma_open_data.py",
    "traffy_fondue.py", "fm91_news.py", "bma_drainage.py",
]

V, D, I, B, TS = "VARCHAR", "DOUBLE", "INTEGER", "BIGINT", "TIMESTAMP"

TABLES = [
    {
        "name": "sun_times", "file": "sun_times/sun_times_daily.csv", "key": ["date"],
        "columns": [("date", "DATE"), ("sunrise", TS), ("sunset", TS), ("daylight_hours", D)],
    },
    {
        "name": "osm_poi", "file": "openstreetmap/osm_poi.csv", "key": ["category", "osm_type", "osm_id"],
        "columns": [("category", V), ("osm_type", V), ("osm_id", B), ("name", V), ("name_en", V),
                    ("network", V), ("latitude", D), ("longitude", D)],
    },
    {
        "name": "osm_roads", "file": "openstreetmap/osm_roads.csv", "key": ["osm_id"],
        "columns": [("osm_id", B), ("road_name", V), ("road_name_en", V), ("highway", V), ("lanes", I),
                    ("oneway", V), ("maxspeed", D), ("ref", V), ("latitude", D), ("longitude", D)],
    },
    {
        "name": "district_population", "file": "bma_open_data/bma_population_district.csv",
        "key": ["district_name"], "where": "TRIM(district_name) <> 'รวม'",
        "columns": [("no", I), ("district_name", V), ("population", B), ("male", B), ("female", B)],
    },
    {
        "name": "district_stats", "file": "bma_open_data/bma_district_stats.csv", "key": ["DISTRICT_CODE"],
        "columns": [("DISTRICT_CODE", I), ("district_name", V), ("DISTRICT_NAME_E", V), ("area_km2", D),
                    ("population", B), ("pop_density_per_km2", D)],
    },
    {
        "name": "district_boundaries", "file": "bma_open_data/bma_districts.geojson", "key": ["DISTRICT_CODE"],
        "columns": [("OBJECTID", I), ("DISTRICT_CODE", I), ("DISTRICT_NAME_T", V), ("DISTRICT_NAME_E", V),
                    ("CHANGWAT_CODE", V), ("CHANGWAT_NAME_T", V), ("CHANGWAT_NAME_E", V),
                    ("SHAPE.AREA", D), ("SHAPE.LEN", D), ("geometry", V)],
    },
    {
        "name": "hawker_zones", "file": "bma_open_data/bma_hawker_zones.csv", "key": ["id"],
        "columns": [("id", I), ("AFiscalYeal", I), ("AreaDist", V), ("AreaName", V), ("num", I), ("Status", V),
                    ("Food_vendors", I), ("Grease_trap_ installers", I), ("latitude", D), ("longitude", D),
                    ("remark", V)],
    },
    {
        "name": "fm91_news", "file": "fm91_news/fm91_news.csv", "key": ["article_id"],
        "columns": [("article_id", B), ("datetime", TS), ("category", V), ("title", V), ("body", V), ("url", V),
                    ("is_traffic_category", I), ("mentions_accident", I), ("mentions_flood", I),
                    ("mentions_rain", I), ("mentions_congestion", I), ("bangkok_related", I),
                    ("roads", V), ("intersections", V)],
    },
    {
        "name": "traffy_complaints", "file": "traffy_fondue/traffy_fondue.csv", "key": ["ticket_id"],
        "columns": [("ticket_id", V), ("timestamp", TS), ("type", V), ("description", V), ("address", V),
                    ("district", V), ("org", V), ("state", V), ("star", D), ("count_reopen", I),
                    ("latitude", D), ("longitude", D), ("photo_url", V), ("after_photo", V), ("note", V),
                    ("last_activity", TS), ("resolve_hours", D), ("is_flood", I), ("is_road", I),
                    ("is_traffic", I), ("is_sidewalk", I)],
    },
    {
        "name": "weather_hourly_points", "file": "open_meteo/weather_hourly_points.csv", "key": ["point", "datetime"],
        "columns": [("point", V), ("latitude", D), ("longitude", D), ("datetime", TS), ("precipitation", D),
                    ("rain", D), ("temperature_2m", D), ("relative_humidity_2m", I), ("cloud_cover", I),
                    ("weather_code", I), ("wind_speed_10m", D), ("is_day", I), ("is_raining", I),
                    ("rain_level", V)],
    },
    {
        "name": "weather_hourly_bangkok", "file": "open_meteo/weather_hourly_bangkok.csv", "key": ["datetime"],
        "columns": [("datetime", TS), ("precipitation_mean", D), ("precipitation_max", D),
                    ("pct_points_raining", D), ("temperature_2m", D), ("relative_humidity_2m", D),
                    ("cloud_cover", D)],
    },
    {
        "name": "metar_observations", "file": "metar/metar_hourly.csv", "key": ["station", "datetime"],
        "columns": [("station", V), ("station_name", V), ("latitude", D), ("longitude", D), ("datetime", TS),
                    ("temp_c", D), ("rh", D), ("wind_kt", D), ("visibility_km", D), ("sky_cover", V),
                    ("wxcodes", V), ("metar", V), ("is_rain", I), ("is_heavy_rain", I),
                    ("is_thunderstorm", I), ("is_shower", I)],
    },
    {
        "name": "drainage_rain_hourly", "file": "bma_drainage/dds_rain_hourly.csv", "key": ["station_code", "datetime"],
        "columns": [("station_code", V), ("district", V), ("station", V), ("datetime", TS), ("rain_mm", D)],
    },
    {
        "name": "drainage_flood_hourly", "file": "bma_drainage/dds_flood_hourly.csv",
        "key": ["station_code", "station", "datetime"],
        "columns": [("station_code", V), ("road", V), ("station", V), ("datetime", TS), ("flood_cm", D)],
    },
    {
        "name": "drainage_canal_level", "file": "bma_drainage/dds_canal_level.csv", "key": ["station_code", "datetime"],
        "columns": [("station_code", V), ("canal", V), ("station", V), ("datetime", TS), ("level_inside_m", D),
                    ("level_outside_m", D), ("level_river_m", D)],
    },
    {
        "name": "drainage_stations", "file": "bma_drainage/dds_stations.csv", "key": ["system", "station_code"],
        "columns": [("system", V), ("station_code", V), ("station", V), ("district_name", V),
                    ("latitude", D), ("longitude", D)],
    },
]


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_scrapers(force: bool = False) -> list:
    failed = []
    for script in SCRAPERS:
        print(f"\n[*] Scraping: {script}", flush=True)
        cmd = [sys.executable, str(WEB_DIR / script)] + (["--force"] if force else [])
        if subprocess.call(cmd, env={**os.environ, "PYTHONUTF8": "1"}) != 0:
            failed.append(script)
            print(f"[x] {script} failed (pipeline continues with existing files)")
    return failed


def geojson_frame(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [{**{k: None if v is None else str(v) for k, v in f["properties"].items()},
             "geometry": json.dumps(f["geometry"], ensure_ascii=False)} for f in data["features"]]
    return pd.DataFrame(rows)


def source_select(path: Path, con) -> str:
    if path.suffix == ".geojson":
        con.register("geojson_source", geojson_frame(path))
        return "SELECT * FROM geojson_source"
    escaped = str(path).replace("\\", "/").replace("'", "''")
    return f"SELECT * FROM read_csv('{escaped}', header=true, all_varchar=true)"


def ingest_bronze(catalog, force: bool = False) -> dict:
    con = catalog.connection
    existing = {name for _, name in catalog.list_tables("bronze")}
    now_ts = datetime.now(timezone.utc).isoformat()
    stats = {}
    for t in TABLES:
        path = RAW_DIR / t["file"]
        bronze = f"{t['name']}_raw"
        if not path.exists():
            print(f"[!] {t['file']} not found (run with --scrape)")
            continue
        sha = sha256(path)
        if bronze in existing and not force:
            if con.execute(f"SELECT count(*) FROM lake.bronze.{bronze} WHERE file_sha256 = ?", [sha]).fetchone()[0]:
                print(f"[!] File guard: {path.name} already in bronze.{bronze} (skipped)")
                continue
        select = (f"SELECT src.*, '{path.name}' AS source_file, "
                  f"CAST(ROW_NUMBER() OVER () AS INTEGER) AS source_row_number, '{sha}' AS file_sha256, "
                  f"CAST('{now_ts}' AS TIMESTAMPTZ) AS ingested_at FROM ({source_select(path, con)}) src")
        with catalog.begin_transaction():
            catalog.set_commit_message(f"bronze: ingest web scraping ({path.name})")
            if bronze in existing:
                con.execute(f"INSERT INTO lake.bronze.{bronze} BY NAME {select}")
            else:
                con.execute(f"CREATE TABLE lake.bronze.{bronze} AS {select}")
                existing.add(bronze)
        n = con.execute(f"SELECT count(*) FROM lake.bronze.{bronze} WHERE file_sha256 = ?", [sha]).fetchone()[0]
        stats[bronze] = n
        print(f"[✓] Ingested {n:,} records into lake.bronze.{bronze}")
    return stats


def typed(column: str, sql_type: str) -> str:
    raw = f"NULLIF(TRIM({quote(column)}), '')"
    if sql_type == V:
        return f"{raw} AS {quote(column)}"
    if sql_type in (I, B):
        return f"CAST(ROUND(TRY_CAST({raw} AS DOUBLE)) AS {sql_type}) AS {quote(column)}"
    return f"TRY_CAST({raw} AS {sql_type}) AS {quote(column)}"


def transform_silver(catalog) -> dict:
    con = catalog.connection
    bronze_tables = {name for _, name in catalog.list_tables("bronze")}
    stats = {}
    for t in TABLES:
        bronze = f"{t['name']}_raw"
        if bronze not in bronze_tables:
            continue
        cols = ",\n    ".join(typed(c, ty) for c, ty in t["columns"])
        key = ", ".join(quote(k) for k in t["key"])
        where = " AND ".join([f"{quote(k)} IS NOT NULL AND TRIM({quote(k)}) <> ''" for k in t["key"]]
                             + ([t["where"]] if t.get("where") else []))
        sql = f"""
            CREATE OR REPLACE TABLE lake.silver.{t['name']} AS
            SELECT {", ".join(quote(c) for c, _ in t["columns"])} FROM (
                SELECT {cols}, source_row_number
                FROM lake.bronze.{bronze}
                WHERE file_sha256 = (SELECT file_sha256 FROM lake.bronze.{bronze} ORDER BY ingested_at DESC LIMIT 1)
                  AND {where}
            )
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY source_row_number DESC) = 1
            ORDER BY {key}
        """
        with catalog.begin_transaction():
            catalog.set_commit_message(f"silver: clean & type {t['name']}")
            con.execute(sql)
        n = con.execute(f"SELECT count(*) FROM lake.silver.{t['name']}").fetchone()[0]
        stats[t["name"]] = n
        print(f"[✓] lake.silver.{t['name']:<24} {n:>8,d} rows")
    return stats


def run(catalog, scrape: bool = False, force_scrape: bool = False, force_ingest: bool = False):
    if scrape or force_scrape:
        print("\n--- [Web Scraping: external sources] ---")
        run_scrapers(force=force_scrape)
    print("\n--- [Web Scraping: Bronze ingestion] ---")
    ingest_bronze(catalog, force=force_ingest)
    print("\n--- [Web Scraping: Silver transformation] ---")
    transform_silver(catalog)
