# Interactive Web Applications (`Apps/`)

Unified multi-page web platform built with Python's `ThreadingHTTPServer`, combining GIS mapping, YOLO11s computer vision, PyDuckLake SQL exploration, and rain-impact analytics.

---

## Running the Web Suite

```bash
# Full server: Map (/), Detect (/inspect), Lakehouse (/lakehouse), and Insights (/insights) on port 8080
python Apps/main.py

# Custom port
python Apps/main.py --port 9000

# Lightweight mode (Insights only, without initializing YOLO or Lakehouse workers)
python Apps/main.py --insights-only
```

### Application Routes

| Route | Module | Purpose |
| :--- | :--- | :--- |
| `/` | `Apps/Camera-Map/` | Interactive Leaflet GIS map with 100 BMA cameras, BTS/MRT route overlays, and historical frame playback. |
| `/inspect` | `Apps/Detection-Classification/` | Fine-tuned YOLO11s single-frame inspector, telemetry HUD, 10-minute cycler, and 16-camera control matrix. |
| `/lakehouse` | `Apps/Lakehouse-Viewer/` | PyDuckLake Bronze/Silver/Gold schema browser, paginated record viewer, CSV export, and read-only SQL workbench. |
| `/insights` | `Apps/Traffic-Insights/` | Rain telemetry findings dashboard, weekday/weekend rain timing audit, camera response filter, and conceptual scorecard. |

---

## Key Modules
- **Camera Map (`Apps/Camera-Map/`):** Uses Leaflet.js across Dark, Streets, Minimal, and Satellite basemaps with bilingual transit overlays (`bangkok_rail.geojson`) and bi-directional deep links (`?camera=<id>&frame=<filename>`). Camera metadata loads directly from `lake.silver.cameras` joined with `lake.silver.camera_transit_proximity`.
- **Detection & Control Room (`Apps/Detection-Classification/`):** Supports keyboard navigation (`←`/`→`, `Space`, `R`), auto-detect timeline playback, 4×4 matrix batch detection, and background batch ingestion into PyDuckLake (`vehicle_counts.csv` $\to$ Bronze $\to$ Silver).
- **Lakehouse Viewer (`Apps/Lakehouse-Viewer/`):** Inspects all 73 tables across `bronze`, `silver`, and `gold` namespaces, executes custom read-only DuckDB SQL queries, and triggers scraper/ETL refreshes.
- **Traffic Insights (`Apps/Traffic-Insights/`):** Queries `lake.gold.rain_*` and `lake.silver.cameras` via `query_lake` to serve `/api/insights` JSON to its SVG UI.

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/map/cameras` & `/api/map/frame` | `GET` | Camera coordinates, transit proximity, and frame existence metadata. |
| `/api/cameras` & `/api/predict` | `GET` / `POST` | Frame indexes and live YOLO11s bounding-box inference. |
| `/api/insights` | `GET` | Preaggregated Gold rain effects, stratified timing, and per-camera Thai/English names from PyDuckLake. |
| `/api/lakehouse/summary`, `/api/lakehouse/table`, `/api/lakehouse/query` | `GET` / `POST` | Live PyDuckLake catalog metrics, table previews, and custom read-only SQL execution. |
| `/images/<camera_id>/<filename>` & `/static/map/<filename>` | `GET` | Raw CCTV JPEG frames and map GeoJSON/icon assets. |
