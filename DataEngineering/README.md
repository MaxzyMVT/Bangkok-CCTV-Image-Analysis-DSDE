# Data Engineering & PyDuckLake Lakehouse (`DataEngineering`)

End-to-end ingestion, data-quality filtering, annotation management, and 3-layer **PyDuckLake Lakehouse** pipeline (`Bronze -> Silver -> Gold`) backing every downstream dashboard, API, and model in the repository.

---

## Architecture: 3-Layer PyDuckLake (`lakehouse/bkk_cctv.ducklake`)

Powered by `DuckLake` (`duckdb` + `pyducklake` catalog metadata + `1,697` Parquet files, `51.87 MB`, `73` tables total across 3 layers) managed through [lakehouse_pipeline.py](lakehouse_pipeline.py):

| Layer | Purpose | Key Tables |
| :--- | :--- | :--- |
| **Bronze** (`24` tables) | Raw ingested CCTV CSVs, frame metadata (`306,595` rows across snapshots), raw vehicle counts (`122,052`), weather, transit, BMA drainage, OSM roads/POIs, and Traffy Fondue CSVs with SHA-256 & row audit columns. | `bronze.cameras_raw`, `bronze.cctv_images_raw`, `bronze.vehicle_counts_raw`, `bronze.transit_stations_raw`, `bronze.weather_hourly_bangkok_raw`, `bronze.traffy_complaints_raw`, `bronze.drainage_flood_hourly_raw` |
| **Silver** (`26` tables) | Strongly typed, deduplicated operational tables and spatial marts ready for analytics and model serving. | `silver.cameras` (`100`), `silver.cctv_frames` (`61,319`), `silver.vehicle_counts` (`60,721`), `silver.counts_v2` (`60,721`), `silver.transit_lines` (`10`), `silver.transit_stations` (`188`), `silver.camera_transit_proximity` (`100`), `silver.sun_times` (`7`), `silver.district_stats` (`50`), `silver.osm_roads` (`19,602`), `silver.osm_poi` (`12,212`), `silver.traffy_complaints` (`7,247`) |
| **Gold** (`23` tables) | Analytical outputs synced by `run_all.py`, `ml_model.py`, and `ingest_analysis_init_tables()`. | `gold.rain_city_effect`, `gold.rain_camera`, `gold.rain_camera_hour`, `gold.rain_camera_effect`, `gold.rain_stratified_effects`, `gold.rain_natural_experiment`, `gold.rain_shelter_*`, `gold.rain_flood_*`, `gold.rain_policy_stations`, `gold.rain_correlations`, `gold.ml_metrics`, `gold.ml_feature_importance`, `gold.ml_predictions`, `gold.rain_ml_*`, `gold.rain_executive_story` |

> **Runtime Rule:** Downstream apps (`DataAnalysis/dashboard.py`, `Apps/Traffic-Insights/app.py`, `Apps/Camera-Map`, `Apps/Lakehouse-Viewer`) query `bkk_cctv.ducklake` directly via `query_lake`, `load_silver_cameras`, `load_silver_sun_times`, `load_gold`, and `DuckLakehouse`. Repository CSV files serve as initialization/rebuild seeds and Power BI Desktop extracts.

---

## Directory Structure

```text
DataEngineering/
├── lakehouse/
│   ├── bkk_cctv.ducklake          # DuckLake metadata catalog
│   └── data/                      # Parquet storage partitioned by bronze/silver/gold
├── quality/
│   ├── find_frozen_frames.py      # Pixel-diff detector isolating frozen/stuck CCTV feeds
│   ├── frozen_frames.csv          # Per-frame frozen audit flags
│   └── frozen_summary.csv         # Per-camera frozen frame statistics
├── labeling/                      # Web-based YOLO pre-labeling & human review app (see labeling/README.md)
├── reviews/                       # Human annotation audit outputs (train/ & test/)
├── train_set/ & test_set/         # Curated YOLO dataset splits
├── WebScraping/                   # 13 external Bangkok collectors & external_lakehouse.py (see WebScraping/README.md)
├── lakehouse_pipeline.py          # Core PyDuckLake builder, query helpers & schema definitions
├── generate_metadata_csv.py       # Rebuilds cctv_images_metadata.csv from image folders & sun times
├── build_rail_geojson.py          # Helper converting transit data to GeoJSON
├── dataset_downloader.py          # Downloads CCTV image dataset
├── cctv_images_metadata.csv       # Seed CSV: 61,319 CCTV frames with spatial & sun-phase metadata
└── vehicle_counts.csv             # Seed CSV: YOLO vehicle counts (car, motorcycle, bus, truck)
```

---

## Core Helpers (`lakehouse_pipeline.py`)

- `query_lake(sql, params=None) -> pd.DataFrame`: Opens an ephemeral DuckDB connection attached to `lake`, executes parameterized SQL, and returns a pandas DataFrame cleanly closing handles.
- `write_lake_table(schema, table, df, message)`: Materializes DataFrames into `lake.<schema>.<table>` with snapshot commit messages.
- `lake_has_table(schema, table) -> bool`: Checks table availability inside `bkk_cctv.ducklake`.
- `ingest_analysis_init_tables()`: Loads `counts_v2.csv` (`silver.counts_v2`) and all `DataAnalysis/output/*.csv` extracts into `gold.rain_*`.
- `load_silver_cameras()` & `load_silver_sun_times()`: Fast Silver lookups with automatic fallback to seed CSVs if the lakehouse has not been built yet.

---

## Commands

```powershell
# Initialize / update Bronze & Silver tables (skipping unchanged files via SHA-256 guard)
python DataEngineering/lakehouse_pipeline.py

# Run missing external WebScraping collectors first, then build Bronze & Silver
python DataEngineering/lakehouse_pipeline.py --scrape

# Full clean rebuild of bkk_cctv.ducklake from scratch
python DataEngineering/lakehouse_pipeline.py --reset

# Rebuild cctv_images_metadata.csv from raw image folders
python DataEngineering/generate_metadata_csv.py

# Detect frozen CCTV camera frames
python DataEngineering/quality/find_frozen_frames.py
```

---

## Sub-Module Documentation

- **External Web Scrapers (Weather, Transit, Floods, Roads, Holidays):** [WebScraping/README.md](WebScraping/README.md)
- **Human-in-the-Loop YOLO Labeling & QA:** [labeling/README.md](labeling/README.md)
