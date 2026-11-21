"""
End-to-End Lakehouse Data Engineering Pipeline for Bangkok CCTV Analysis
Implements a two-layer Lakehouse architecture (Bronze -> Silver) using PyDuckLake & Parquet.

Architecture:
  Raw Sources:
    1. BMA CCTV Cameras Info & Image Metadata (61,319 frames across 100 cameras)
    2. Bangkok Rapid Transit Web Scraping (BTS / MRT / ARL / SRT routes & stations via OSM Overpass)
  Bronze Layer:
    - Raw text ingestion preserving exact source records with metadata audit columns:
      source_file, source_row_number, file_sha256, ingested_at.
    - Tables: bronze.cameras_raw, bronze.cctv_images_raw, bronze.transit_lines_raw, bronze.transit_stations_raw
  Silver Layer:
    - Cleaned, validated, strongly typed, deduplicated tables stored as Parquet files.
    - Tables: silver.cameras, silver.cctv_frames, silver.transit_lines, silver.transit_stations
    - Spatial Proximity Mart: silver.camera_transit_proximity (computes geodesic distances
      from every CCTV surveillance node to nearest rapid transit stations).

Usage:
    python DataEngineering/lakehouse_pipeline.py
    python DataEngineering/lakehouse_pipeline.py --reset
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Ensure pyducklake imports cleanly
try:
    from pyducklake import (
        Catalog,
        Schema,
        StringType,
        IntegerType,
        BigIntType,
        DoubleType,
        BooleanType,
        TimestampTZType,
        DateType,
        optional,
        required,
    )
except ImportError:
    # If invoked with system Python, redirect to .venv if available
    venv_py = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists() and sys.executable != str(venv_py):
        import subprocess
        sys.exit(subprocess.call([str(venv_py), __file__] + sys.argv[1:]))
    raise

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_ENG_DIR = ROOT_DIR / "DataEngineering"
LAKEHOUSE_DIR = DATA_ENG_DIR / "lakehouse"
CATALOG_PATH = LAKEHOUSE_DIR / "bkk_cctv.ducklake"
PARQUET_DATA_DIR = LAKEHOUSE_DIR / "data"

sys.path.insert(0, str(DATA_ENG_DIR / "WebScraping"))
import external_lakehouse  # noqa: E402

CCTV_INFO_CSV = ROOT_DIR / "dataset" / "100_cameras" / "cctv_info.csv"
CCTV_METADATA_CSV = DATA_ENG_DIR / "cctv_images_metadata.csv"
RAIL_GEOJSON = ROOT_DIR / "Apps" / "Camera-Map" / "static" / "bangkok_rail.geojson"
VEHICLE_COUNTS_CSV = DATA_ENG_DIR / "vehicle_counts.csv"


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for idempotency file guarding."""
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# -------------------------------------------------------------------------
# Schemas: Bronze Layer (Raw text + Audit columns)
# -------------------------------------------------------------------------
BRONZE_CAMERAS_SCHEMA = Schema.of(
    optional("camera_id", StringType()),
    optional("location_th", StringType()),
    optional("direction_th", StringType()),
    optional("location_en", StringType()),
    optional("location_detail_en", StringType()),
    optional("latitude", StringType()),
    optional("longitude", StringType()),
    optional("ip_address", StringType()),
    required("source_file", StringType()),
    required("source_row_number", IntegerType()),
    required("file_sha256", StringType()),
    required("ingested_at", TimestampTZType()),
)

BRONZE_IMAGES_SCHEMA = Schema.of(
    optional("image_id", StringType()),
    optional("camera_id", StringType()),
    optional("filename", StringType()),
    optional("relative_path", StringType()),
    optional("location_th", StringType()),
    optional("direction_th", StringType()),
    optional("location_en", StringType()),
    optional("location_detail_en", StringType()),
    optional("latitude", StringType()),
    optional("longitude", StringType()),
    optional("ip_address", StringType()),
    optional("timestamp", StringType()),
    optional("date", StringType()),
    optional("time", StringType()),
    optional("year", StringType()),
    optional("month", StringType()),
    optional("day", StringType()),
    optional("day_of_week", StringType()),
    optional("is_weekend", StringType()),
    optional("hour", StringType()),
    optional("minute", StringType()),
    optional("second", StringType()),
    optional("time_period", StringType()),
    optional("width", StringType()),
    optional("height", StringType()),
    optional("aspect_ratio", StringType()),
    optional("file_size_bytes", StringType()),
    required("source_file", StringType()),
    required("source_row_number", IntegerType()),
    required("file_sha256", StringType()),
    required("ingested_at", TimestampTZType()),
)

BRONZE_LINES_SCHEMA = Schema.of(
    optional("line_key", StringType()),
    optional("system", StringType()),
    optional("name_th", StringType()),
    optional("name_en", StringType()),
    optional("colour", StringType()),
    optional("geometry_type", StringType()),
    optional("coordinate_count", StringType()),
    required("source_file", StringType()),
    required("source_row_number", IntegerType()),
    required("file_sha256", StringType()),
    required("ingested_at", TimestampTZType()),
)

BRONZE_STATIONS_SCHEMA = Schema.of(
    optional("name_th", StringType()),
    optional("name_en", StringType()),
    optional("lines_json", StringType()),
    optional("latitude", StringType()),
    optional("longitude", StringType()),
    required("source_file", StringType()),
    required("source_row_number", IntegerType()),
    required("file_sha256", StringType()),
    required("ingested_at", TimestampTZType()),
)

BRONZE_VEHICLE_COUNTS_SCHEMA = Schema.of(
    optional("image_file", StringType()),
    optional("camera_id", StringType()),
    optional("timestamp", StringType()),
    optional("date", StringType()),
    optional("hour", StringType()),
    optional("minute", StringType()),
    optional("latitude", StringType()),
    optional("longitude", StringType()),
    optional("location_en", StringType()),
    optional("total_vehicles", StringType()),
    optional("car", StringType()),
    optional("motorcycle", StringType()),
    optional("bus", StringType()),
    optional("truck", StringType()),
    optional("bicycle", StringType()),
    required("source_file", StringType()),
    required("source_row_number", IntegerType()),
    required("file_sha256", StringType()),
    required("ingested_at", TimestampTZType()),
)


# -------------------------------------------------------------------------
# Schemas: Silver Layer (Strongly typed, validated, enriched)
# -------------------------------------------------------------------------
SILVER_CAMERAS_SCHEMA = Schema.of(
    required("camera_id", StringType()),
    required("location_th", StringType()),
    optional("direction_th", StringType()),
    optional("location_en", StringType()),
    optional("location_detail_en", StringType()),
    required("latitude", DoubleType()),
    required("longitude", DoubleType()),
    optional("ip_address", StringType()),
)

SILVER_FRAMES_SCHEMA = Schema.of(
    required("image_id", StringType()),
    required("camera_id", StringType()),
    required("filename", StringType()),
    required("captured_at", TimestampTZType()),
    required("capture_date", DateType()),
    required("hour", IntegerType()),
    required("time_period", StringType()),
    required("is_weekend", BooleanType()),
    required("width", IntegerType()),
    required("height", IntegerType()),
    required("file_size_bytes", BigIntType()),
)

SILVER_LINES_SCHEMA = Schema.of(
    required("line_key", StringType()),
    required("system", StringType()),
    required("name_th", StringType()),
    required("name_en", StringType()),
    required("colour", StringType()),
)

SILVER_STATIONS_SCHEMA = Schema.of(
    required("station_id", StringType()),
    required("name_th", StringType()),
    optional("name_en", StringType()),
    required("connected_lines", StringType()),
    required("primary_system", StringType()),
    required("latitude", DoubleType()),
    required("longitude", DoubleType()),
    required("is_interchange", BooleanType()),
)

SILVER_PROXIMITY_SCHEMA = Schema.of(
    required("camera_id", StringType()),
    required("camera_name_th", StringType()),
    required("camera_lat", DoubleType()),
    required("camera_lon", DoubleType()),
    required("nearest_station_th", StringType()),
    optional("nearest_station_en", StringType()),
    required("transit_system", StringType()),
    required("distance_meters", DoubleType()),
    required("is_near_transit", BooleanType()),
)

SILVER_VEHICLE_COUNTS_SCHEMA = Schema.of(
    required("image_file", StringType()),
    required("camera_id", StringType()),
    required("captured_at", TimestampTZType()),
    required("capture_date", DateType()),
    required("hour", IntegerType()),
    required("minute", IntegerType()),
    required("total_vehicles", IntegerType()),
    required("car", IntegerType()),
    required("motorcycle", IntegerType()),
    required("bus", IntegerType()),
    required("truck", IntegerType()),
    required("bicycle", IntegerType()),
)


# -------------------------------------------------------------------------
# Lakehouse Core Engine
# -------------------------------------------------------------------------
class DuckLakehouse:
    def __init__(self, lakehouse_dir: Path = LAKEHOUSE_DIR, reset: bool = False):
        self.lakehouse_dir = lakehouse_dir
        self.catalog_path = lakehouse_dir / "bkk_cctv.ducklake"
        self.data_path = lakehouse_dir / "data"

        if reset and self.lakehouse_dir.exists():
            print(f"[*] Resetting lakehouse directory: {self.lakehouse_dir}")
            shutil.rmtree(self.lakehouse_dir, ignore_errors=True)

        self.lakehouse_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize PyDuckLake Catalog
        self.catalog = Catalog(
            "lake",
            str(self.catalog_path),
            data_path=str(self.data_path) + "/"
        )
        # Disable inlining so every commit generates physical Parquet files
        self.catalog.set_option("data_inlining_row_limit", "0")
        self.catalog.connection.execute("SET TimeZone = 'Asia/Bangkok'")

    def initialize_tables(self):
        """Creates bronze and silver namespaces and tables in a single ACID transaction."""
        existing_namespaces = set(self.catalog.list_namespaces())
        with self.catalog.begin_transaction():
            self.catalog.set_commit_message("init: create bronze and silver namespaces & tables")
            if "bronze" not in existing_namespaces:
                self.catalog.create_namespace("bronze")
            if "silver" not in existing_namespaces:
                self.catalog.create_namespace("silver")

            bronze_tables = {name for _, name in self.catalog.list_tables("bronze")}
            silver_tables = {name for _, name in self.catalog.list_tables("silver")}

            # Bronze Tables
            if "cameras_raw" not in bronze_tables:
                self.catalog.create_table(("bronze", "cameras_raw"), BRONZE_CAMERAS_SCHEMA)
            if "cctv_images_raw" not in bronze_tables:
                self.catalog.create_table(("bronze", "cctv_images_raw"), BRONZE_IMAGES_SCHEMA)
            if "transit_lines_raw" not in bronze_tables:
                self.catalog.create_table(("bronze", "transit_lines_raw"), BRONZE_LINES_SCHEMA)
            if "transit_stations_raw" not in bronze_tables:
                self.catalog.create_table(("bronze", "transit_stations_raw"), BRONZE_STATIONS_SCHEMA)
            if "vehicle_counts_raw" not in bronze_tables:
                self.catalog.create_table(("bronze", "vehicle_counts_raw"), BRONZE_VEHICLE_COUNTS_SCHEMA)

            # Silver Tables
            if "cameras" not in silver_tables:
                self.catalog.create_table(("silver", "cameras"), SILVER_CAMERAS_SCHEMA)
            if "cctv_frames" not in silver_tables:
                self.catalog.create_table(("silver", "cctv_frames"), SILVER_FRAMES_SCHEMA)
            if "transit_lines" not in silver_tables:
                self.catalog.create_table(("silver", "transit_lines"), SILVER_LINES_SCHEMA)
            if "transit_stations" not in silver_tables:
                self.catalog.create_table(("silver", "transit_stations"), SILVER_STATIONS_SCHEMA)
            if "camera_transit_proximity" not in silver_tables:
                self.catalog.create_table(("silver", "camera_transit_proximity"), SILVER_PROXIMITY_SCHEMA)
            if "vehicle_counts" not in silver_tables:
                self.catalog.create_table(("silver", "vehicle_counts"), SILVER_VEHICLE_COUNTS_SCHEMA)

        print("[✓] Initialized Bronze & Silver schemas in PyDuckLake catalog")

    def ingest_bronze(self, force: bool = False) -> Dict[str, int]:
        """Ingests raw CSV and GeoJSON feeds into bronze tables with file guard protection."""
        stats = {}
        now_ts = datetime.now(timezone.utc).isoformat()
        con = self.catalog.connection

        # 1. Ingest cctv_info.csv
        if CCTV_INFO_CSV.exists():
            sha = compute_sha256(CCTV_INFO_CSV)
            fname = CCTV_INFO_CSV.name
            already_loaded = False
            if not force:
                chk = con.execute("SELECT count(*) FROM lake.bronze.cameras_raw WHERE file_sha256 = ?", [sha]).fetchone()[0]
                already_loaded = chk > 0

            if already_loaded:
                print(f"[!] File guard: {fname} already ingested into bronze.cameras_raw (skipped)")
                stats["cameras_raw"] = 0
            else:
                with open(CCTV_INFO_CSV, "r", encoding="cp874", errors="replace") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                with self.catalog.begin_transaction():
                    self.catalog.set_commit_message(f"bronze: ingest raw cameras ({fname})")
                    for idx, row in enumerate(rows, start=1):
                        con.execute("""
                            INSERT INTO lake.bronze.cameras_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            row.get("camera_id"),
                            row.get("location"),
                            row.get("location_detail"),
                            row.get("location_en"),
                            row.get("location_detail_en"),
                            row.get("latitude"),
                            row.get("longitude"),
                            row.get("ip_address"),
                            fname,
                            idx,
                            sha,
                            now_ts
                        ])
                cnt = con.execute("SELECT count(*) FROM lake.bronze.cameras_raw WHERE file_sha256 = ?", [sha]).fetchone()[0]
                stats["cameras_raw"] = cnt
                print(f"[✓] Ingested {cnt} records into lake.bronze.cameras_raw")

        # 2. Ingest cctv_images_metadata.csv
        if CCTV_METADATA_CSV.exists():
            sha = compute_sha256(CCTV_METADATA_CSV)
            fname = CCTV_METADATA_CSV.name
            already_loaded = False
            if not force:
                chk = con.execute("SELECT count(*) FROM lake.bronze.cctv_images_raw WHERE file_sha256 = ?", [sha]).fetchone()[0]
                already_loaded = chk > 0

            if already_loaded:
                print(f"[!] File guard: {fname} already ingested into bronze.cctv_images_raw (skipped)")
                stats["cctv_images_raw"] = 0
            else:
                csv_path_escaped = str(CCTV_METADATA_CSV).replace("\\", "/")
                with self.catalog.begin_transaction():
                    self.catalog.set_commit_message(f"bronze: ingest raw image metadata ({fname})")
                    con.execute(f"""
                        INSERT INTO lake.bronze.cctv_images_raw
                        SELECT
                            image_id, camera_id, filename, relative_path,
                            location_th, direction_th, location_en, location_detail_en,
                            CAST(latitude AS VARCHAR), CAST(longitude AS VARCHAR), ip_address,
                            timestamp, date, time, year, month, day, day_of_week,
                            is_weekend, hour, minute, second, time_period,
                            CAST(width AS VARCHAR), CAST(height AS VARCHAR),
                            CAST(aspect_ratio AS VARCHAR), CAST(file_size_bytes AS VARCHAR),
                            '{fname}' AS source_file,
                            ROW_NUMBER() OVER () AS source_row_number,
                            '{sha}' AS file_sha256,
                            CAST('{now_ts}' AS TIMESTAMPTZ) AS ingested_at
                        FROM read_csv('{csv_path_escaped}', header=true, all_varchar=true)
                    """)
                cnt = con.execute("SELECT count(*) FROM lake.bronze.cctv_images_raw WHERE file_sha256 = ?", [sha]).fetchone()[0]
                stats["cctv_images_raw"] = cnt
                print(f"[✓] Ingested {cnt} records into lake.bronze.cctv_images_raw")

        # 3. Ingest bangkok_rail.geojson (BTS / MRT / ARL / SRT scraping)
        if RAIL_GEOJSON.exists():
            sha = compute_sha256(RAIL_GEOJSON)
            fname = RAIL_GEOJSON.name
            already_loaded = False
            if not force:
                chk = con.execute("SELECT count(*) FROM lake.bronze.transit_lines_raw WHERE file_sha256 = ?", [sha]).fetchone()[0]
                already_loaded = chk > 0

            if already_loaded:
                print(f"[!] File guard: {fname} already ingested into transit tables (skipped)")
                stats["transit_lines_raw"] = 0
                stats["transit_stations_raw"] = 0
            else:
                with open(RAIL_GEOJSON, "r", encoding="utf-8") as f:
                    geojson_data = json.load(f)

                line_features = [f for f in geojson_data.get("features", []) if f.get("properties", {}).get("kind") == "line"]
                station_features = [f for f in geojson_data.get("features", []) if f.get("properties", {}).get("kind") == "station"]

                with self.catalog.begin_transaction():
                    self.catalog.set_commit_message(f"bronze: ingest transit scraping ({fname})")
                    # Ingest lines
                    for idx, lf in enumerate(line_features, start=1):
                        lp = lf.get("properties", {})
                        coords = lf.get("geometry", {}).get("coordinates", [])
                        con.execute("""
                            INSERT INTO lake.bronze.transit_lines_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            lp.get("key"), lp.get("system"), lp.get("name_th"), lp.get("name_en"),
                            lp.get("colour"), lf.get("geometry", {}).get("type"), str(len(coords)),
                            fname, idx, sha, now_ts
                        ])

                    # Ingest stations
                    for idx, sf in enumerate(station_features, start=1):
                        sp = sf.get("properties", {})
                        coord = sf.get("geometry", {}).get("coordinates", [0, 0])
                        con.execute("""
                            INSERT INTO lake.bronze.transit_stations_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            sp.get("name_th"), sp.get("name_en"), json.dumps(sp.get("lines", []), ensure_ascii=False),
                            str(coord[1]), str(coord[0]),
                            fname, idx, sha, now_ts
                        ])

                stats["transit_lines_raw"] = len(line_features)
                stats["transit_stations_raw"] = len(station_features)
                print(f"[✓] Ingested {len(line_features)} transit lines and {len(station_features)} stations into bronze")

        # 4. Ingest vehicle_counts.csv if it exists
        if VEHICLE_COUNTS_CSV.exists():
            sha = compute_sha256(VEHICLE_COUNTS_CSV)
            fname = VEHICLE_COUNTS_CSV.name
            already_loaded = False
            if not force:
                chk = con.execute("SELECT count(*) FROM lake.bronze.vehicle_counts_raw WHERE file_sha256 = ?", [sha]).fetchone()[0]
                already_loaded = chk > 0

            if already_loaded:
                print(f"[!] File guard: {fname} already ingested into bronze.vehicle_counts_raw (skipped)")
            else:
                escaped = str(VEHICLE_COUNTS_CSV).replace("\\", "/").replace("'", "''")
                with self.catalog.begin_transaction():
                    self.catalog.set_commit_message(f"bronze: ingest vehicle counts ({fname})")
                    con.execute(f"""
                        DELETE FROM lake.bronze.vehicle_counts_raw WHERE file_sha256 = '{sha}';
                        INSERT INTO lake.bronze.vehicle_counts_raw
                        SELECT
                            image_file,
                            camera_id,
                            timestamp,
                            date,
                            hour,
                            minute,
                            latitude,
                            longitude,
                            location_en,
                            total_vehicles,
                            car,
                            motorcycle,
                            bus,
                            truck,
                            bicycle,
                            '{fname}' AS source_file,
                            CAST(ROW_NUMBER() OVER () AS INTEGER) AS source_row_number,
                            '{sha}' AS file_sha256,
                            CAST('{now_ts}' AS TIMESTAMPTZ) AS ingested_at
                        FROM read_csv('{escaped}', header=true, all_varchar=true);
                    """)
                cnt = con.execute("SELECT count(*) FROM lake.bronze.vehicle_counts_raw WHERE file_sha256 = ?", [sha]).fetchone()[0]
                stats["vehicle_counts_raw"] = cnt
                print(f"[✓] Ingested {cnt:,} vehicle detection records into bronze.vehicle_counts_raw")

        return stats

    def transform_silver(self) -> Dict[str, int]:
        """Cleans, validates, types, and enriches data from bronze into silver layer."""
        stats = {}
        con = self.catalog.connection

        # 1. Clean cameras
        with self.catalog.begin_transaction():
            self.catalog.set_commit_message("silver: clean & deduplicate camera stations")
            con.execute("""
                DELETE FROM lake.silver.cameras;
                INSERT INTO lake.silver.cameras
                SELECT
                    TRIM(camera_id) AS camera_id,
                    TRIM(location_th) AS location_th,
                    NULLIF(TRIM(direction_th), '') AS direction_th,
                    NULLIF(TRIM(location_en), '') AS location_en,
                    NULLIF(TRIM(location_detail_en), '') AS location_detail_en,
                    TRY_CAST(latitude AS DOUBLE) AS latitude,
                    TRY_CAST(longitude AS DOUBLE) AS longitude,
                    NULLIF(TRIM(ip_address), '') AS ip_address
                FROM lake.bronze.cameras_raw
                WHERE camera_id IS NOT NULL
                  AND TRY_CAST(latitude AS DOUBLE) IS NOT NULL
                  AND TRY_CAST(longitude AS DOUBLE) IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY camera_id ORDER BY ingested_at DESC) = 1;
            """)
        c_cnt = con.execute("SELECT count(*) FROM lake.silver.cameras").fetchone()[0]
        stats["cameras"] = c_cnt

        # 2. Clean cctv_frames
        with self.catalog.begin_transaction():
            self.catalog.set_commit_message("silver: type and validate cctv frames")
            con.execute("""
                DELETE FROM lake.silver.cctv_frames;
                INSERT INTO lake.silver.cctv_frames
                SELECT
                    TRIM(image_id) AS image_id,
                    TRIM(camera_id) AS camera_id,
                    TRIM(filename) AS filename,
                    TRY_CAST(timestamp AS TIMESTAMPTZ) AS captured_at,
                    TRY_CAST(date AS DATE) AS capture_date,
                    TRY_CAST(hour AS INTEGER) AS hour,
                    COALESCE(NULLIF(TRIM(time_period), ''), 'Midday') AS time_period,
                    CASE WHEN LOWER(TRIM(is_weekend)) IN ('true', '1') THEN TRUE ELSE FALSE END AS is_weekend,
                    TRY_CAST(width AS INTEGER) AS width,
                    TRY_CAST(height AS INTEGER) AS height,
                    TRY_CAST(file_size_bytes AS BIGINT) AS file_size_bytes
                FROM lake.bronze.cctv_images_raw
                WHERE image_id IS NOT NULL
                  AND TRY_CAST(timestamp AS TIMESTAMPTZ) IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY image_id ORDER BY ingested_at DESC) = 1;
            """)
        f_cnt = con.execute("SELECT count(*) FROM lake.silver.cctv_frames").fetchone()[0]
        stats["cctv_frames"] = f_cnt

        # 3. Clean transit lines
        with self.catalog.begin_transaction():
            self.catalog.set_commit_message("silver: clean transit lines")
            con.execute("""
                DELETE FROM lake.silver.transit_lines;
                INSERT INTO lake.silver.transit_lines
                SELECT
                    TRIM(line_key) AS line_key,
                    UPPER(TRIM(system)) AS system,
                    TRIM(name_th) AS name_th,
                    TRIM(name_en) AS name_en,
                    TRIM(colour) AS colour
                FROM lake.bronze.transit_lines_raw
                WHERE line_key IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY line_key ORDER BY ingested_at DESC) = 1;
            """)
        l_cnt = con.execute("SELECT count(*) FROM lake.silver.transit_lines").fetchone()[0]
        stats["transit_lines"] = l_cnt

        # 4. Clean transit stations
        with self.catalog.begin_transaction():
            self.catalog.set_commit_message("silver: clean transit stations")
            con.execute("""
                DELETE FROM lake.silver.transit_stations;
                INSERT INTO lake.silver.transit_stations
                SELECT
                    MD5(CONCAT(TRIM(name_th), '_', latitude, '_', longitude)) AS station_id,
                    TRIM(name_th) AS name_th,
                    NULLIF(TRIM(name_en), '') AS name_en,
                    lines_json AS connected_lines,
                    CASE
                        WHEN lines_json LIKE '%bts%' THEN 'BTS'
                        WHEN lines_json LIKE '%mrt%' THEN 'MRT'
                        WHEN lines_json LIKE '%arl%' THEN 'ARL'
                        WHEN lines_json LIKE '%srt%' THEN 'SRT'
                        ELSE 'OTHER'
                    END AS primary_system,
                    TRY_CAST(latitude AS DOUBLE) AS latitude,
                    TRY_CAST(longitude AS DOUBLE) AS longitude,
                    CASE WHEN lines_json LIKE '%,%' THEN TRUE ELSE FALSE END AS is_interchange
                FROM lake.bronze.transit_stations_raw
                WHERE name_th IS NOT NULL
                  AND TRY_CAST(latitude AS DOUBLE) IS NOT NULL
                  AND TRY_CAST(longitude AS DOUBLE) IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY TRIM(name_th) ORDER BY ingested_at DESC) = 1;
            """)
        s_cnt = con.execute("SELECT count(*) FROM lake.silver.transit_stations").fetchone()[0]
        stats["transit_stations"] = s_cnt

        # 5. Build Spatial Mart: silver.camera_transit_proximity
        # Uses Haversine spherical distance formula on coordinates to find nearest rapid transit station
        with self.catalog.begin_transaction():
            self.catalog.set_commit_message("silver: calculate camera-transit geodesic proximity")
            con.execute("""
                DELETE FROM lake.silver.camera_transit_proximity;
                WITH paired AS (
                    SELECT
                        c.camera_id,
                        c.location_th AS camera_name_th,
                        c.latitude AS camera_lat,
                        c.longitude AS camera_lon,
                        s.name_th AS nearest_station_th,
                        s.name_en AS nearest_station_en,
                        s.primary_system AS transit_system,
                        -- Haversine formula in meters
                        ROUND(6371000 * 2 * ASIN(SQRT(
                            POWER(SIN(RADIANS(s.latitude - c.latitude) / 2), 2) +
                            COS(RADIANS(c.latitude)) * COS(RADIANS(s.latitude)) *
                            POWER(SIN(RADIANS(s.longitude - c.longitude) / 2), 2)
                        )), 1) AS distance_meters
                    FROM lake.silver.cameras c
                    CROSS JOIN lake.silver.transit_stations s
                ),
                ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (PARTITION BY camera_id ORDER BY distance_meters ASC) AS rank
                    FROM paired
                )
                INSERT INTO lake.silver.camera_transit_proximity
                SELECT
                    camera_id,
                    camera_name_th,
                    camera_lat,
                    camera_lon,
                    nearest_station_th,
                    nearest_station_en,
                    transit_system,
                    distance_meters,
                    CASE WHEN distance_meters <= 500 THEN TRUE ELSE FALSE END AS is_near_transit
                FROM ranked
                WHERE rank = 1;
            """)
        p_cnt = con.execute("SELECT count(*) FROM lake.silver.camera_transit_proximity").fetchone()[0]
        stats["camera_transit_proximity"] = p_cnt

        # 6. Clean and type vehicle counts if bronze exists and has data
        v_cnt = 0
        has_vc = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'bronze' AND table_name = 'vehicle_counts_raw'").fetchone()[0] > 0
        if has_vc and con.execute("SELECT count(*) FROM lake.bronze.vehicle_counts_raw").fetchone()[0] > 0:
            with self.catalog.begin_transaction():
                self.catalog.set_commit_message("silver: clean, type & deduplicate vehicle counts")
                con.execute("""
                    DELETE FROM lake.silver.vehicle_counts;
                    INSERT INTO lake.silver.vehicle_counts
                    SELECT
                        image_file,
                        camera_id,
                        TRY_CAST(timestamp AS TIMESTAMPTZ) AS captured_at,
                        TRY_CAST(date AS DATE) AS capture_date,
                        TRY_CAST(hour AS INTEGER) AS hour,
                        TRY_CAST(minute AS INTEGER) AS minute,
                        COALESCE(TRY_CAST(total_vehicles AS INTEGER), 0) AS total_vehicles,
                        COALESCE(TRY_CAST(car AS INTEGER), 0) AS car,
                        COALESCE(TRY_CAST(motorcycle AS INTEGER), 0) AS motorcycle,
                        COALESCE(TRY_CAST(bus AS INTEGER), 0) AS bus,
                        COALESCE(TRY_CAST(truck AS INTEGER), 0) AS truck,
                        COALESCE(TRY_CAST(bicycle AS INTEGER), 0) AS bicycle
                    FROM lake.bronze.vehicle_counts_raw
                    WHERE image_file IS NOT NULL
                      AND camera_id IS NOT NULL
                      AND TRY_CAST(timestamp AS TIMESTAMPTZ) IS NOT NULL
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY camera_id, image_file ORDER BY ingested_at DESC) = 1;
                """)
            v_cnt = con.execute("SELECT count(*) FROM lake.silver.vehicle_counts").fetchone()[0]
            stats["vehicle_counts"] = v_cnt

        print(f"[✓] Silver transformation complete: {c_cnt} cameras, {f_cnt} frames, {l_cnt} transit lines, {s_cnt} stations, {p_cnt} proximity links, {v_cnt} vehicle counts")
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Queries table statistics and parquet file footprints."""
        con = self.catalog.connection
        summary = {
            "catalog": str(self.catalog_path),
            "parquet_files": len(list(self.data_path.glob("**/*.parquet"))),
            "total_size_mb": round(sum(f.stat().st_size for f in self.data_path.glob("**/*.parquet")) / (1024 * 1024), 2),
            "tables": {}
        }
        for ns in ["bronze", "silver"]:
            for _, tname in self.catalog.list_tables(ns):
                cnt = con.execute(f"SELECT count(*) FROM lake.{ns}.{tname}").fetchone()[0]
                summary["tables"][f"{ns}.{tname}"] = cnt
        return summary


def run_pipeline(reset: bool = False, force: bool = False, scrape: bool = False, rescrape: bool = False):
    print("=" * 68)
    print(" BANGKOK CCTV DATA ENGINEERING: PYDUCKLAKE LAKEHOUSE PIPELINE")
    print("=" * 68)

    # 1. Step 1: Ensure metadata CSV is generated
    if not CCTV_METADATA_CSV.exists():
        print("[*] Generating master CCTV images metadata catalog...")
        import subprocess
        subprocess.check_call([sys.executable, str(DATA_ENG_DIR / "generate_metadata_csv.py")])

    # 2. Step 2: Ensure mass transit GeoJSON is up-to-date
    if not RAIL_GEOJSON.exists():
        print("[*] Extracting Bangkok transit routes from OpenStreetMap...")
        import subprocess
        subprocess.check_call([sys.executable, str(DATA_ENG_DIR / "WebScraping" / "build_rail_geojson.py")])

    # 3. Step 3: Initialize Lakehouse
    lake = DuckLakehouse(LAKEHOUSE_DIR, reset=reset)
    lake.initialize_tables()

    # 4. Step 4: Bronze Ingestion
    print("\n--- [Phase 1: Bronze Layer Ingestion] ---")
    bronze_stats = lake.ingest_bronze(force=force)

    # 5. Step 5: Silver Layer Transformation & Proximity Mart
    print("\n--- [Phase 2: Silver Layer Transformation & Spatial Enrichment] ---")
    silver_stats = lake.transform_silver()

    # 6. Step 6: External web scraping sources (scrape -> bronze -> silver)
    external_lakehouse.run(lake.catalog, scrape=scrape, force_scrape=rescrape, force_ingest=force)

    # 7. Step 7: Pipeline Inspection & Report
    print("\n--- [Phase 3: Lakehouse Audit & Summary] ---")
    summary = lake.get_summary()
    print(f"Catalog DB:     {summary['catalog']}")
    print(f"Parquet Files:  {summary['parquet_files']} files ({summary['total_size_mb']} MB)")
    print("\nRow Counts:")
    for tbl, count in summary["tables"].items():
        print(f"  {tbl:<30} {count:>8,d} rows")

    # Sample query on the spatial proximity mart
    print("\n--- Sample Insight: Cameras within 500m of BTS/MRT Stations ---")
    con = lake.catalog.connection
    sample = con.execute("""
        SELECT
            camera_id, camera_name_th, nearest_station_th, transit_system, distance_meters
        FROM lake.silver.camera_transit_proximity
        WHERE is_near_transit = TRUE
        ORDER BY distance_meters ASC
        LIMIT 5;
    """).fetchall()
    for row in sample:
        print(f"  Cam {row[0]:<4} ({row[1][:22]}...) -> {row[2]} ({row[3]}) | {row[4]}m away")

    print("\n[✓] End-to-end Data Engineering pipeline executed successfully!")


# -------------------------------------------------------------------------
# Lakehouse Reader Helpers (Single Source of Truth for Web Apps & Models)
# -------------------------------------------------------------------------
def load_silver_cameras() -> Dict[str, Dict[str, Any]]:
    """Loads camera metadata and rapid transit proximity from lake.silver.cameras.
    Falls back to CCTV_INFO_CSV if lakehouse is not initialized.
    """
    if CATALOG_PATH.exists() and PARQUET_DATA_DIR.exists():
        try:
            import duckdb
            cat_posix = CATALOG_PATH.resolve().as_posix()
            data_posix = PARQUET_DATA_DIR.resolve().as_posix() + "/"
            con = duckdb.connect()
            con.execute(f"ATTACH '{cat_posix}' AS lake (TYPE DUCKLAKE, DATA_PATH '{data_posix}')")
            sql = """
                SELECT 
                    c.camera_id,
                    c.location_th,
                    c.direction_th,
                    c.location_en,
                    c.location_detail_en,
                    c.latitude,
                    c.longitude,
                    c.ip_address,
                    p.nearest_station_th,
                    p.nearest_station_en,
                    p.transit_system,
                    p.distance_meters,
                    p.is_near_transit
                FROM lake.silver.cameras c
                LEFT JOIN lake.silver.camera_transit_proximity p ON c.camera_id = p.camera_id
            """
            rows = con.execute(sql).df().to_dict(orient="records")
            con.close()
            res = {}
            for r in rows:
                cid = str(r["camera_id"])
                r["location_detail_th"] = r.get("direction_th") or ""
                # Provide standard keys for backward compatibility
                r["location"] = r.get("location_th") or ""
                r["location_detail"] = r.get("direction_th") or ""
                res[cid] = r
            if res:
                return res
        except Exception as e:
            print(f"[Lakehouse Reader] Warning querying lake.silver.cameras: {e}. Falling back to CSV.")

    from Models.pipeline import load_cctv_metadata
    return load_cctv_metadata(CCTV_INFO_CSV)


def load_silver_sun_times() -> Dict[str, Any]:
    """Loads date -> (sunrise_minute, sunset_minute) from lake.silver.sun_times.
    Falls back to raw sun_times_daily.csv if lakehouse is not initialized.
    """
    if CATALOG_PATH.exists() and PARQUET_DATA_DIR.exists():
        try:
            import duckdb
            cat_posix = CATALOG_PATH.resolve().as_posix()
            data_posix = PARQUET_DATA_DIR.resolve().as_posix() + "/"
            con = duckdb.connect()
            con.execute(f"ATTACH '{cat_posix}' AS lake (TYPE DUCKLAKE, DATA_PATH '{data_posix}')")
            df = con.execute("SELECT date, sunrise, sunset FROM lake.silver.sun_times ORDER BY date").df()
            con.close()
            lookup = {}
            for _, r in df.iterrows():
                d_str = str(r["date"]).split(" ")[0].strip()
                sr, ss = r["sunrise"], r["sunset"]
                sr_min = sr.hour * 60 + sr.minute
                ss_min = ss.hour * 60 + ss.minute
                lookup[d_str] = (sr_min, ss_min)
            if lookup:
                return lookup
        except Exception as e:
            print(f"[Lakehouse Reader] Warning querying lake.silver.sun_times: {e}. Falling back to CSV.")
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangkok CCTV PyDuckLake Lakehouse Pipeline")
    parser.add_argument("--reset", action="store_true", help="Recreate lakehouse database from scratch")
    parser.add_argument("--force", action="store_true", help="Force re-ingest all files bypassing file guards")
    parser.add_argument("--scrape", action="store_true", help="Run web scrapers for missing raw files before ingestion")
    parser.add_argument("--rescrape", action="store_true", help="Re-run every web scraper even if raw files exist")
    args = parser.parse_args()

    run_pipeline(reset=args.reset, force=args.force, scrape=args.scrape, rescrape=args.rescrape)
