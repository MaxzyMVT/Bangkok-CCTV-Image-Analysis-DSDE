"""
Bangkok CCTV web app: Camera Map + Detect on one server.

    python Apps/main.py            # http://localhost:8080
    python Apps/main.py --port 9000

Routes:
    /                    Camera Map         (Apps/Camera-Map)
    /inspect             Detect / YOLO      (Apps/Detection-Classification)
    /api/map/...         Map data           (/api/map/cameras, /api/map/frame)
    /images/<cam>/<file> Map snapshots
    /api/...             Detect API         (/api/cameras, /api/image, /api/predict, ...)

Frames are identified in URLs by camera + filename, for example
    /?camera=915&frame=915_..._20260831_210012.jpg
    /inspect?camera=915&frame=915_..._20260831_210012.jpg
"""
import sys
import argparse
import threading
import importlib.util
import urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer

APPS_DIR = Path(__file__).resolve().parent
ROOT_DIR = APPS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Startup messages include symbols like "✓"; don't crash on a cp1252 Windows console
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Folder names contain dashes, so load the apps by file path
detect_app = _load_module("detect_app", APPS_DIR / "Detection-Classification" / "app.py")
map_app = _load_module("camera_map_app", APPS_DIR / "Camera-Map" / "app.py")
lakehouse_app = _load_module("lakehouse_app", APPS_DIR / "Lakehouse-Viewer" / "app.py")

# One YOLO model is shared by all requests; run inference one request at a time
PREDICT_LOCK = threading.Lock()


class CombinedHandler(detect_app.CCTVAnalysisHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        query = urllib.parse.parse_qs(url.query)

        # Normalization redirects
        if path == "/inspect/":
            self.send_response(301)
            self.send_header("Location", "/inspect" + (f"?{url.query}" if url.query else ""))
            self.end_headers()
            return
        if path == "/lakehouse/":
            self.send_response(301)
            self.send_header("Location", "/lakehouse" + (f"?{url.query}" if url.query else ""))
            self.end_headers()
            return

        # 1. Lakehouse Viewer routes (/lakehouse, /api/lakehouse/*)
        if lakehouse_app.handle_lakehouse_get(self, path, query):
            return

        # 2. Map routes ("/", "/api/map/*", "/images/*", "/static/map/*")
        if map_app.handle_map_get(self, path, query):
            return

        # 3. Detect & YOLO routes
        super().do_GET()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)

        # 1. Lakehouse Viewer POST endpoints (/api/lakehouse/query, /api/lakehouse/pipeline/run)
        if lakehouse_app.handle_lakehouse_post(self, path):
            return

        # 2. YOLO Predict & detection POST endpoints (protected by PREDICT_LOCK)
        with PREDICT_LOCK:
            super().do_POST()

    def log_message(self, fmt, *args):
        pass


class SilentThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        exc_type, _, _ = sys.exc_info()
        if exc_type in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            return
        super().handle_error(request, client_address)


def ensure_lakehouse():
    """
    Auto-initializes the DuckLakehouse on server startup if not yet created.
    If the catalog or silver tables are missing or empty:
      1. Initializes Bronze and Silver schemas & tables
      2. Ingests raw data from git (vehicle_counts.csv, cctv_info.csv, scraped weather/holidays/transit)
      3. Transforms Silver tables (cleaning, proximity mart, temporal features)
    This ensures that anyone cloning the repository has an immediately functional Lakehouse out-of-the-box!
    """
    try:
        from DataEngineering.lakehouse_pipeline import DuckLakehouse, CATALOG_PATH
        needs_init = not CATALOG_PATH.exists()
        if not needs_init:
            try:
                lake = DuckLakehouse(reset=False)
                res = lake.catalog.connection.execute("SELECT COUNT(*) FROM lake.silver.cameras").fetchone()
                if not res or res[0] == 0:
                    needs_init = True
            except Exception:
                needs_init = True

        if needs_init:
            print("[Lakehouse] Catalog not found or empty. Auto-bootstrapping Lakehouse from repository raw data...")
            lake = DuckLakehouse(reset=False)
            lake.initialize_tables()
            bronze_stats = lake.ingest_bronze()
            silver_stats = lake.transform_silver()

            # Also sync external scraped tables (BMA drainage, weather, OSM roads, population, Traffy)
            try:
                sys.path.insert(0, str(ROOT_DIR / "DataEngineering" / "WebScraping"))
                import external_lakehouse
                ext_b = external_lakehouse.ingest_bronze(lake.catalog, force=False)
                ext_s = external_lakehouse.transform_silver(lake.catalog)
                print(f"[Lakehouse] External tables synced! Bronze: {len(ext_b)}, Silver: {len(ext_s)}")
            except Exception as ext_err:
                print(f"[Lakehouse] Note: External tables sync warning: {ext_err}")

            print(f"[Lakehouse] Auto-bootstrap complete! Bronze: {bronze_stats}, Silver: {silver_stats}")
        else:
            print("[Lakehouse] PyDuckLake catalog verified and ready.")
    except Exception as e:
        print(f"[Lakehouse] Note: Auto-bootstrap check skipped: {e}")


def start_server(port=8080):
    # Same startup as the Detect app: fetch dataset if missing, load metadata, index frames
    detect_app.ensure_dataset()
    if not detect_app.METADATA_LOOKUP and detect_app.CCTV_INFO_CSV.exists():
        detect_app.METADATA_LOOKUP = detect_app.load_cctv_metadata(detect_app.CCTV_INFO_CSV)
    detect_app.build_camera_index()
    map_app.load_cameras()

    # Auto-initialize Lakehouse if tables or catalog are missing
    ensure_lakehouse()

    # Pre-warm YOLO detector and PyTorch on main thread to avoid multi-thread import race
    try:
        import torch
        import torch._utils
        print("[App] Pre-warming YOLO model on main thread...")
        detector = detect_app.get_detector()
        _ = detector.model
        print("[App] YOLO model loaded and ready.")
    except Exception as e:
        print(f"[!] Warning pre-warming YOLO detector: {e}")

    server = SilentThreadingHTTPServer(("", port), CombinedHandler)
    active_model = detect_app.resolve_model()
    print("=" * 64)
    print(" Bangkok CCTV: Camera Map + YOLO11 Detect + Lakehouse Explorer")
    print(f" Model:     {active_model}")
    print(f" Cameras:   {len(map_app.CAMERAS)}")
    print(f" Map:       http://localhost:{port}/")
    print(f" Detect:    http://localhost:{port}/inspect")
    print(f" Lakehouse: http://localhost:{port}/lakehouse")
    print(" Press Ctrl+C to stop")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangkok CCTV Camera Map + Detect server")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    args = parser.parse_args()
    start_server(port=args.port)
