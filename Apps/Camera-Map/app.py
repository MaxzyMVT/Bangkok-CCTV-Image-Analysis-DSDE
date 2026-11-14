"""
Bangkok CCTV Camera Map - lightweight web server.

Serves an interactive Leaflet + OpenStreetMap map of Bangkok with a pin
for each CCTV camera in dataset/100_cameras/cctv_info.csv.
Uses only the Python standard library (no extra installs needed).

Usage:
    python Apps/Camera-Map/app.py --port 8081

Endpoints:
    GET /                              Map page
    GET /api/cameras                   All cameras (id, names, lat/lon, image count, latest image)
    GET /images/<camera_id>/<file>     A CCTV snapshot from the dataset
"""
import csv
import json
import argparse
import mimetypes
import urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATE_FILE = Path(__file__).resolve().parent / "templates" / "index.html"
DATASET_DIR = ROOT_DIR / "dataset" / "100_cameras" / "cctv_images"
CCTV_INFO_CSV = ROOT_DIR / "dataset" / "100_cameras" / "cctv_info.csv"

# Placeholder English names in the scraped data that should fall back to Thai
PLACEHOLDER_NAMES = {"", "connect bmacctv"}


def _clean(value):
    value = (value or "").strip()
    return "" if value.lower() in PLACEHOLDER_NAMES else value


def load_cameras():
    """Reads camera metadata and pairs each camera with its image folder."""
    cameras = []
    if not CCTV_INFO_CSV.exists():
        print(f"[!] Camera metadata not found: {CCTV_INFO_CSV}")
        return cameras

    with open(CCTV_INFO_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, ValueError):
                continue

            cam_id = row["camera_id"].strip()
            cam_dir = DATASET_DIR / cam_id
            images = sorted(p.name for p in cam_dir.glob("*.jpg")) if cam_dir.is_dir() else []

            cameras.append({
                "camera_id": cam_id,
                "name_en": _clean(row.get("location_en")),
                "detail_en": _clean(row.get("location_detail_en")),
                "name_th": _clean(row.get("location")),
                "detail_th": _clean(row.get("location_detail")),
                "latitude": lat,
                "longitude": lon,
                "image_count": len(images),
                "first_image": images[0] if images else None,
                "latest_image": images[-1] if images else None,
            })
    return cameras


CAMERAS = []


class MapRequestHandler(BaseHTTPRequestHandler):
    def _send_bytes(self, body, content_type, status=200, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.unquote(self.path.split("?", 1)[0])

        if path in ("/", "/index.html"):
            self._send_bytes(TEMPLATE_FILE.read_bytes(), "text/html; charset=utf-8")

        elif path == "/api/cameras":
            body = json.dumps({"cameras": CAMERAS, "total": len(CAMERAS)}, ensure_ascii=False)
            self._send_bytes(body.encode("utf-8"), "application/json; charset=utf-8")

        elif path.startswith("/images/"):
            self._serve_image(path[len("/images/"):])

        else:
            self.send_error(404, "Not found")

    def _serve_image(self, rel):
        # Only allow files that live inside the dataset image folder
        target = (DATASET_DIR / rel).resolve()
        try:
            target.relative_to(DATASET_DIR.resolve())
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        if not target.is_file():
            self.send_error(404, "Image not found")
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), ctype, cache=True)

    def log_message(self, fmt, *args):
        # Keep the console quiet except for errors
        pass


def start_server(port=8081):
    global CAMERAS
    CAMERAS = load_cameras()
    with_images = sum(1 for c in CAMERAS if c["image_count"])

    server = ThreadingHTTPServer(("", port), MapRequestHandler)
    print("=" * 60)
    print(" Bangkok CCTV Camera Map")
    print(f" Cameras loaded: {len(CAMERAS)} ({with_images} with images)")
    print(f" Local URL: http://localhost:{port}")
    print(" Press Ctrl+C to stop")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangkok CCTV Camera Map server")
    parser.add_argument("--port", type=int, default=8081, help="Web server port (default: 8081)")
    args = parser.parse_args()
    start_server(port=args.port)
