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


# Folder names contain dashes, so load the two apps by file path
detect_app = _load_module("detect_app", APPS_DIR / "Detection-Classification" / "app.py")
map_app = _load_module("camera_map_app", APPS_DIR / "Camera-Map" / "app.py")

# One YOLO model is shared by all requests; run inference one request at a time
PREDICT_LOCK = threading.Lock()


class CombinedHandler(detect_app.CCTVAnalysisHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        query = urllib.parse.parse_qs(url.query)

        # Map owns "/", "/api/map/*" and "/images/*"; everything else goes to Detect
        if path == "/inspect/":
            self.send_response(301)
            self.send_header("Location", "/inspect" + (f"?{url.query}" if url.query else ""))
            self.end_headers()
            return
        if map_app.handle_map_get(self, path, query):
            return
        super().do_GET()

    def do_POST(self):
        with PREDICT_LOCK:
            super().do_POST()

    def log_message(self, fmt, *args):
        pass


def start_server(port=8080):
    # Same startup as the Detect app: fetch dataset if missing, load metadata, index frames
    detect_app.ensure_dataset()
    if not detect_app.METADATA_LOOKUP and detect_app.CCTV_INFO_CSV.exists():
        detect_app.METADATA_LOOKUP = detect_app.load_cctv_metadata(detect_app.CCTV_INFO_CSV)
    detect_app.build_camera_index()
    map_app.load_cameras()

    server = ThreadingHTTPServer(("", port), CombinedHandler)
    print("=" * 64)
    print(" Bangkok CCTV: Camera Map + Detect")
    print(f" Cameras: {len(map_app.CAMERAS)}")
    print(f" Map:     http://localhost:{port}/")
    print(f" Detect:  http://localhost:{port}/inspect")
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
