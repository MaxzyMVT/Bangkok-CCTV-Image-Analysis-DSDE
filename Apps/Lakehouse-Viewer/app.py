"""
Bangkok CCTV Lakehouse Viewer & Data Explorer Application
Provides schema inspection, live table browsing, SQL query console, and data pipeline re-ingestion.

Can run standalone:
    python Apps/Lakehouse-Viewer/app.py --port 8082
Or integrated into Apps/main.py on port 8080 at /lakehouse.
"""
import os
import sys
import json
import math
import csv
import io
import time
import argparse
import threading
import subprocess
import urllib.parse
from pathlib import Path
from datetime import datetime, date, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Startup console encoding safety
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DataEngineering.lakehouse_pipeline import DuckLakehouse, LAKEHOUSE_DIR

TEMPLATE_FILE = Path(__file__).resolve().parent / "templates" / "index.html"
SCRAPING_DIR = ROOT_DIR / "DataEngineering" / "WebScraping"
RAIL_SCRIPT = SCRAPING_DIR / "build_rail_geojson.py"

# Global lock for DuckLakehouse access
LAKEHOUSE_LOCK = threading.Lock()

# Pipeline background worker state
PIPELINE_STATE = {
    "running": False,
    "mode": "idle",
    "stage": "idle",
    "percent": 0.0,
    "message": "Pipeline idle",
    "logs": [],
    "error": None,
    "stats": None
}
PIPELINE_LOCK = threading.Lock()


def _log_pipeline(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(f"[Lakehouse Pipeline] {entry}")
    with PIPELINE_LOCK:
        PIPELINE_STATE["logs"].append(entry)
        if len(PIPELINE_STATE["logs"]) > 200:
            PIPELINE_STATE["logs"] = PIPELINE_STATE["logs"][-200:]
        PIPELINE_STATE["message"] = msg


def _serialize_val(v):
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4) if abs(v) < 1e12 else v
    if isinstance(v, (bytes, bytearray)):
        return f"<bytes:{len(v)}>"
    return v


def get_table_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("vehicle", "camera", "frame", "cctv")):
        return "CCTV & Vehicle Detections"
    if any(k in n for k in ("transit", "station", "line", "proximity", "rail")):
        return "Rapid Transit & Spatial Hubs"
    if any(k in n for k in ("weather", "metar", "sun", "rain", "precipitation")):
        return "Weather & Atmospheric Conditions"
    if any(k in n for k in ("drainage", "flood", "canal")):
        return "BMA Drainage & Flood Sensors"
    if any(k in n for k in ("district", "population", "hawker", "traffy", "fm91", "osm")):
        return "Urban & Municipal Context"
    return "General Tables"


def get_lakehouse_summary():
    with LAKEHOUSE_LOCK:
        lake = DuckLakehouse()
        con = lake.catalog.connection
        namespaces = lake.catalog.list_namespaces()
        
        tables_list = []
        total_rows = 0
        
        for ns in ["silver", "bronze"]:
            if ns not in namespaces:
                continue
            for _, tname in lake.catalog.list_tables(ns):
                try:
                    cnt = con.execute(f"SELECT COUNT(*) FROM lake.{ns}.{tname}").fetchone()[0]
                except Exception:
                    cnt = 0
                total_rows += cnt
                
                # Column count
                try:
                    col_info = con.execute(f"DESCRIBE lake.{ns}.{tname}").fetchall()
                    col_count = len(col_info)
                except Exception:
                    col_count = 0
                    
                tables_list.append({
                    "schema": ns,
                    "name": tname,
                    "full_name": f"{ns}.{tname}",
                    "row_count": cnt,
                    "column_count": col_count,
                    "category": get_table_category(tname)
                })

        data_path = lake.data_path
        parquet_files = list(data_path.glob("**/*.parquet"))
        total_bytes = sum(f.stat().st_size for f in parquet_files) if parquet_files else 0
        total_mb = round(total_bytes / (1024 * 1024), 2)
        
        return {
            "catalog_path": str(lake.catalog_path),
            "parquet_files": len(parquet_files),
            "total_size_mb": total_mb,
            "total_records": total_rows,
            "total_tables": len(tables_list),
            "namespaces": namespaces,
            "tables": tables_list
        }


def get_table_schema(schema: str, table: str):
    schema = schema.strip().lower()
    table = table.strip().lower()
    if schema not in ("bronze", "silver"):
        raise ValueError("Invalid schema: must be bronze or silver")

    with LAKEHOUSE_LOCK:
        lake = DuckLakehouse()
        con = lake.catalog.connection
        # Verify table exists in catalog
        valid_tables = {tname for _, tname in lake.catalog.list_tables(schema)}
        if table not in valid_tables:
            raise ValueError(f"Table '{schema}.{table}' does not exist in catalog")

        desc_rows = con.execute(f"DESCRIBE lake.{schema}.{table}").fetchall()
        columns = []
        for r in desc_rows:
            col_name = r[0]
            col_type = r[1]
            nullable = r[2] == "YES"
            columns.append({
                "name": col_name,
                "type": col_type,
                "nullable": nullable,
                "default": r[4] if len(r) > 4 else None
            })

        count = con.execute(f"SELECT COUNT(*) FROM lake.{schema}.{table}").fetchone()[0]

        return {
            "schema": schema,
            "table": table,
            "full_name": f"{schema}.{table}",
            "row_count": count,
            "columns": columns
        }


def get_table_data(schema: str, table: str, page: int = 1, page_size: int = 50, search: str = "", sort_col: str = "", sort_dir: str = "ASC"):
    schema = schema.strip().lower()
    table = table.strip().lower()
    if schema not in ("bronze", "silver"):
        raise ValueError("Invalid schema: must be bronze or silver")

    page = max(1, page)
    page_size = min(max(5, page_size), 500)

    with LAKEHOUSE_LOCK:
        lake = DuckLakehouse()
        con = lake.catalog.connection
        valid_tables = {tname for _, tname in lake.catalog.list_tables(schema)}
        if table not in valid_tables:
            raise ValueError(f"Table '{schema}.{table}' does not exist in catalog")

        desc = con.execute(f"DESCRIBE lake.{schema}.{table}").fetchall()
        columns = [r[0] for r in desc]
        string_cols = [r[0] for r in desc if "VARCHAR" in r[1].upper() or "TEXT" in r[1].upper()]

        where_clause = ""
        params = []
        if search:
            search_str = f"%{search.strip()}%"
            or_parts = []
            for sc in string_cols[:6]:  # search in up to 6 string columns
                or_parts.append(f'"{sc}" ILIKE ?')
                params.append(search_str)
            if or_parts:
                where_clause = "WHERE (" + " OR ".join(or_parts) + ")"

        # Count total matching rows
        count_sql = f"SELECT COUNT(*) FROM lake.{schema}.{table} {where_clause}"
        total_rows = con.execute(count_sql, params).fetchone()[0]

        # Sorting
        order_clause = ""
        if sort_col and sort_col in columns:
            direction = "DESC" if sort_dir.upper() == "DESC" else "ASC"
            order_clause = f'ORDER BY "{sort_col}" {direction} NULLS LAST'
        elif "captured_at" in columns:
            order_clause = 'ORDER BY "captured_at" DESC'
        elif "ingested_at" in columns:
            order_clause = 'ORDER BY "ingested_at" DESC'
        elif "camera_id" in columns:
            order_clause = 'ORDER BY "camera_id" ASC'

        offset = (page - 1) * page_size
        query_sql = f'SELECT * FROM lake.{schema}.{table} {where_clause} {order_clause} LIMIT {page_size} OFFSET {offset}'
        raw_rows = con.execute(query_sql, params).fetchall()

        rows = []
        for r in raw_rows:
            rows.append({col: _serialize_val(val) for col, val in zip(columns, r)})

        total_pages = math.ceil(total_rows / page_size) if total_rows > 0 else 1

        return {
            "schema": schema,
            "table": table,
            "columns": columns,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "rows": rows
        }


def execute_custom_query(sql_text: str):
    sql = sql_text.strip()
    if not sql:
        raise ValueError("Empty SQL query")

    # Guard: only allow safe read-only queries
    upper_sql = sql.upper().strip()
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "CREATE TABLE", "CREATE SCHEMA"]
    for keyword in forbidden:
        if keyword in upper_sql:
            raise ValueError(f"Disallowed DDL/DML operation in explorer mode: '{keyword}'")

    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH") or upper_sql.startswith("DESCRIBE") or upper_sql.startswith("EXPLAIN") or upper_sql.startswith("SHOW")):
        raise ValueError("Only read-only SELECT, WITH, or DESCRIBE queries are supported")

    start_time = time.perf_counter()
    with LAKEHOUSE_LOCK:
        lake = DuckLakehouse()
        con = lake.catalog.connection
        # Enforce max 1,000 rows
        if "LIMIT" not in upper_sql:
            sql += " LIMIT 1000"

        cur = con.execute(sql)
        desc = cur.description or []
        columns = [d[0] for d in desc]
        raw_rows = cur.fetchall()

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    rows = []
    for r in raw_rows:
        rows.append({col: _serialize_val(val) for col, val in zip(columns, r)})

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "execution_ms": duration_ms
    }


def export_table_csv(schema: str, table: str, limit: int = 5000) -> str:
    schema = schema.strip().lower()
    table = table.strip().lower()
    with LAKEHOUSE_LOCK:
        lake = DuckLakehouse()
        con = lake.catalog.connection
        valid_tables = {tname for _, tname in lake.catalog.list_tables(schema)}
        if table not in valid_tables:
            raise ValueError(f"Table '{schema}.{table}' does not exist")

        cur = con.execute(f"SELECT * FROM lake.{schema}.{table} LIMIT {limit}")
        desc = cur.description
        columns = [d[0] for d in desc]
        rows = cur.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    for r in rows:
        writer.writerow([_serialize_val(v) if v is not None else "" for v in r])
    return output.getvalue()


def _run_pipeline_worker(mode: str = "ingest_only"):
    global PIPELINE_STATE
    try:
        with PIPELINE_LOCK:
            PIPELINE_STATE["running"] = True
            PIPELINE_STATE["mode"] = mode
            PIPELINE_STATE["stage"] = "init"
            PIPELINE_STATE["percent"] = 5.0
            PIPELINE_STATE["message"] = f"Initializing pipeline worker (mode={mode})..."
            PIPELINE_STATE["logs"] = []
            PIPELINE_STATE["error"] = None
            PIPELINE_STATE["stats"] = None

        _log_pipeline(f"Starting lakehouse pipeline execution [Mode: {mode}]")

        # Step 1: Rapid Transit GeoJSON Scraping
        if mode in ("scrape_rail", "full_scrape"):
            _log_pipeline("Executing Overpass scraper: build_rail_geojson.py (Bangkok Transit Lines & Stations)...")
            with PIPELINE_LOCK:
                PIPELINE_STATE["stage"] = "scraping_rail"
                PIPELINE_STATE["percent"] = 20.0

            py_exe = sys.executable
            proc = subprocess.run(
                [py_exe, str(RAIL_SCRIPT)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            if proc.returncode == 0:
                _log_pipeline("✓ OpenStreetMap railway GeoJSON built successfully for Camera Map & Lakehouse!")
            else:
                _log_pipeline(f"⚠ Warning: build_rail_geojson.py error: {proc.stderr[:160]}")

        # Step 2: External Web Scrapers
        if mode == "full_scrape":
            _log_pipeline("Running external web scrapers (Open-Meteo, METAR, Sun Times, OSM)...")
            with PIPELINE_LOCK:
                PIPELINE_STATE["stage"] = "scraping_external"
                PIPELINE_STATE["percent"] = 40.0

            sys.path.insert(0, str(SCRAPING_DIR))
            import external_lakehouse
            external_lakehouse.run_scrapers(force=False)
            _log_pipeline("✓ Web scraping feeds refreshed!")

        # Step 3: Lakehouse Ingestion & Transformation
        with PIPELINE_LOCK:
            PIPELINE_STATE["stage"] = "lakehouse_ingest"
            PIPELINE_STATE["percent"] = 65.0
        _log_pipeline("Ingesting Bronze layer records with SHA-256 file audit guards...")

        with LAKEHOUSE_LOCK:
            lake = DuckLakehouse(LAKEHOUSE_DIR, reset=False)
            lake.initialize_tables()
            bronze_stats = lake.ingest_bronze(force=True)
            _log_pipeline(f"✓ Ingested Bronze tables: {list(bronze_stats.keys())}")

            with PIPELINE_LOCK:
                PIPELINE_STATE["stage"] = "lakehouse_transform"
                PIPELINE_STATE["percent"] = 85.0
            _log_pipeline("Executing Silver transformations, data typing & camera-transit proximity mart...")

            silver_stats = lake.transform_silver()
            _log_pipeline(f"✓ Transformed Silver tables: {list(silver_stats.keys())}")

            # Also run external_lakehouse bronze & silver
            sys.path.insert(0, str(SCRAPING_DIR))
            import external_lakehouse
            _log_pipeline("Synchronizing external scraped tables into Bronze & Silver layers...")
            ext_b = external_lakehouse.ingest_bronze(lake.catalog, force=True)
            ext_s = external_lakehouse.transform_silver(lake.catalog)
            _log_pipeline(f"✓ External tables synced (Bronze: {len(ext_b)}, Silver: {len(ext_s)})")

        with PIPELINE_LOCK:
            PIPELINE_STATE["running"] = False
            PIPELINE_STATE["stage"] = "complete"
            PIPELINE_STATE["percent"] = 100.0
            PIPELINE_STATE["message"] = "Pipeline completed successfully!"
            PIPELINE_STATE["stats"] = {
                "bronze": {**bronze_stats, **ext_b},
                "silver": {**silver_stats, **ext_s}
            }
        _log_pipeline("🎉 Pipeline finished! Medallion lakehouse and Camera Map railway layer are fully populated.")

    except Exception as e:
        import traceback
        err_msg = str(e)
        _log_pipeline(f"❌ Pipeline failed: {err_msg}")
        with PIPELINE_LOCK:
            PIPELINE_STATE["running"] = False
            PIPELINE_STATE["stage"] = "error"
            PIPELINE_STATE["error"] = err_msg
            PIPELINE_STATE["message"] = f"Error: {err_msg}"


def _send(handler, content: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200, headers: dict = None):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    if headers:
        for k, v in headers.items():
            handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(content)


def _send_json(handler, data, status: int = 200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    _send(handler, body, "application/json; charset=utf-8", status=status)


def handle_lakehouse_get(handler, path: str, query: dict) -> bool:
    """Routes GET requests for the Lakehouse Viewer. Returns True if handled."""
    # 1. Main UI
    if path in ("/lakehouse", "/lakehouse/", "/lakehouse/index.html"):
        if TEMPLATE_FILE.exists():
            _send(handler, TEMPLATE_FILE.read_bytes(), "text/html; charset=utf-8")
        else:
            handler.send_error(404, "Lakehouse template not found")
        return True

    # 2. Lakehouse Summary & Table Inventory
    elif path == "/api/lakehouse/summary":
        try:
            summary = get_lakehouse_summary()
            _send_json(handler, summary)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, status=500)
        return True

    # 3. Table Schema & Column Metadata
    elif path == "/api/lakehouse/schema":
        schema = query.get("schema", ["silver"])[0]
        table = query.get("table", [""])[0]
        if not table:
            _send_json(handler, {"error": "Missing table parameter"}, status=400)
            return True
        try:
            res = get_table_schema(schema, table)
            _send_json(handler, res)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, status=400)
        return True

    # 4. Table Paginated Data View
    elif path == "/api/lakehouse/data":
        schema = query.get("schema", ["silver"])[0]
        table = query.get("table", [""])[0]
        page = int(query.get("page", [1])[0])
        page_size = int(query.get("page_size", [50])[0])
        search = query.get("search", [""])[0]
        sort_col = query.get("sort_col", [""])[0]
        sort_dir = query.get("sort_dir", ["ASC"])[0]

        if not table:
            _send_json(handler, {"error": "Missing table parameter"}, status=400)
            return True
        try:
            data = get_table_data(schema, table, page=page, page_size=page_size, search=search, sort_col=sort_col, sort_dir=sort_dir)
            _send_json(handler, data)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, status=500)
        return True

    # 5. Export Table Data as CSV
    elif path == "/api/lakehouse/export":
        schema = query.get("schema", ["silver"])[0]
        table = query.get("table", [""])[0]
        limit = int(query.get("limit", [5000])[0])
        if not table:
            _send_json(handler, {"error": "Missing table parameter"}, status=400)
            return True
        try:
            csv_text = export_table_csv(schema, table, limit=limit)
            csv_bytes = csv_text.encode("utf-8-sig")
            filename = f"lakehouse_{schema}_{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            _send(handler, csv_bytes, "text/csv; charset=utf-8", headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            })
        except Exception as e:
            _send_json(handler, {"error": str(e)}, status=500)
        return True

    # 6. Pipeline Background Status & Logs
    elif path == "/api/lakehouse/pipeline/status":
        with PIPELINE_LOCK:
            st = dict(PIPELINE_STATE)
            st["logs"] = list(PIPELINE_STATE["logs"])
        _send_json(handler, st)
        return True

    return False


def handle_lakehouse_post(handler, path: str) -> bool:
    """Routes POST requests for the Lakehouse Viewer. Returns True if handled."""
    content_len = int(handler.headers.get("Content-Length", 0))
    raw_body = handler.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
    try:
        req_data = json.loads(raw_body)
    except Exception:
        req_data = {}

    # 1. Custom SQL Query Execution
    if path == "/api/lakehouse/query":
        sql = req_data.get("query", "")
        if not sql:
            _send_json(handler, {"error": "Missing SQL query string"}, status=400)
            return True
        try:
            res = execute_custom_query(sql)
            _send_json(handler, res)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, status=400)
        return True

    # 2. Trigger Pipeline Background Re-Ingest / Web Scraping
    elif path == "/api/lakehouse/pipeline/run":
        with PIPELINE_LOCK:
            if PIPELINE_STATE["running"]:
                _send_json(handler, {"error": "Pipeline already running", "state": PIPELINE_STATE}, status=409)
                return True
        mode = req_data.get("mode", "ingest_only")
        t = threading.Thread(target=_run_pipeline_worker, args=(mode,), daemon=True)
        t.start()
        _send_json(handler, {"status": "started", "mode": mode, "message": f"Lakehouse pipeline started in mode '{mode}'"})
        return True

    return False


class StandaloneLakehouseHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            path = "/lakehouse"
        if not handle_lakehouse_get(self, path, query):
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        if not handle_lakehouse_post(self, path):
            self.send_error(404, "Endpoint not found")

    def log_message(self, fmt, *args):
        pass


def start_server(port: int = 8082):
    server = ThreadingHTTPServer(("", port), StandaloneLakehouseHandler)
    print("=" * 64)
    print(f" 🏛️ Bangkok CCTV Lakehouse Explorer App Running!")
    print(f" 👉 URL: http://localhost:{port}/lakehouse")
    print(f" Press Ctrl+C to stop")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Lakehouse server.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangkok CCTV Lakehouse Viewer")
    parser.add_argument("--port", type=int, default=8082, help="Port (default: 8082)")
    args = parser.parse_args()
    start_server(port=args.port)
