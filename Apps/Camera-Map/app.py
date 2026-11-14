"""
Bangkok CCTV Camera Map - map data, image routes and a standalone server.

The combined server (Apps/main.py) imports this module and serves the map at "/".
This file can also run on its own, without the Detect page:
    python Apps/Camera-Map/app.py --port 8081

Endpoints:
    GET /                                   Map page
    GET /api/map/cameras                    All cameras (names, lat/lon or null, image count, latest frame)
    GET /api/map/frame?camera_id=&filename= Whether a frame exists for that camera (+ its date/time)
    GET /images/<camera_id>/<filename>      A CCTV snapshot from the dataset

Uses only the Python standard library.
"""
import re
import csv
import json
import math
import argparse
import mimetypes
import urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATE_FILE = Path(__file__).resolve().parent / "templates" / "index.html"
DATASET_DIR = ROOT_DIR / "dataset" / "100_cameras" / "cctv_images"
CCTV_INFO_CSV = ROOT_DIR / "dataset" / "100_cameras" / "cctv_info.csv"

# Same format as Models/pipeline.py: <camera_id>_<description>_<YYYYMMDD>_<HHMMSS>.jpg
FILENAME_PATTERN = re.compile(r"^(\d+)_(.+)_(\d{8})_(\d{6})\.jpe?g$", re.IGNORECASE)

# Placeholder values in the scraped data that should count as "missing"
PLACEHOLDER_NAMES = {"", "0", "-", "nan", "none", "connect bmacctv"}

CAMERAS = []          # filled by load_cameras() at startup
CAMERA_FILES = {}     # camera_id -> set of image filenames


def _clean(value):
    value = (value or "").strip()
    return "" if value.lower() in PLACEHOLDER_NAMES else value


def _coord(value, low, high):
    """Float coordinate inside [low, high], else None (missing / zero / garbage)."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or num == 0 or not (low <= num <= high):
        return None
    return num


def frame_stamp(filename):
    """'..._20260831_210012.jpg' -> {'date': '2026-08-31', 'time': '21:00:12'} or None."""
    m = FILENAME_PATTERN.match(filename or "")
    if not m:
        return None
    d, t = m.group(3), m.group(4)
    return {"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "time": f"{t[:2]}:{t[2:4]}:{t[4:]}"}


def frame_belongs_to_camera(camera_id, filename):
    if not camera_id or not filename or "/" in filename or "\\" in filename:
        return False
    m = FILENAME_PATTERN.match(filename)
    return bool(m) and m.group(1) == str(camera_id)


def resolve_frame(camera_id, filename):
    """Absolute path of a real frame of camera_id inside cctv_images, else None."""
    if not frame_belongs_to_camera(camera_id, filename):
        return None
    try:
        full = (DATASET_DIR / str(camera_id) / filename).resolve()
        full.relative_to(DATASET_DIR.resolve())
    except (ValueError, OSError):
        return None
    return full if full.is_file() else None


def load_cameras():
    """Reads camera metadata and pairs each camera with its image folder."""
    global CAMERAS, CAMERA_FILES
    cameras, files_by_cam = [], {}
    if not CCTV_INFO_CSV.exists():
        print(f"[!] Camera metadata not found: {CCTV_INFO_CSV}")
    else:
        with open(CCTV_INFO_CSV, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                cam_id = (row.get("camera_id") or "").strip()
                if not cam_id:
                    continue
                cam_dir = DATASET_DIR / cam_id
                images = sorted(p.name for p in cam_dir.glob("*.jpg")) if cam_dir.is_dir() else []
                files_by_cam[cam_id] = set(images)

                # Bangkok sits around 13.7N, 100.5E; accept anything plausible for Thailand
                lat = _coord(row.get("latitude"), 5.0, 21.0)
                lon = _coord(row.get("longitude"), 97.0, 106.0)
                has_coords = lat is not None and lon is not None

                cameras.append({
                    "camera_id": cam_id,
                    "name_en": _clean(row.get("location_en")),
                    "detail_en": _clean(row.get("location_detail_en")),
                    "name_th": _clean(row.get("location")),
                    "detail_th": _clean(row.get("location_detail")),
                    "latitude": lat if has_coords else None,
                    "longitude": lon if has_coords else None,
                    "has_coords": has_coords,
                    "image_count": len(images),
                    "first_image": images[0] if images else None,
                    "latest_image": images[-1] if images else None,
                })
    CAMERAS, CAMERA_FILES = cameras, files_by_cam
    return cameras


def cameras_payload():
    return {"cameras": CAMERAS, "total": len(CAMERAS)}


def frame_payload(camera_id, filename):
    """Answer for /api/map/frame: does this frame exist for this camera?"""
    known = any(c["camera_id"] == camera_id for c in CAMERAS)
    found = (known and filename in CAMERA_FILES.get(camera_id, set())
             and resolve_frame(camera_id, filename) is not None)
    return {
        "camera_id": camera_id,
        "filename": filename,
        "camera_found": known,
        "found": bool(found),
        "stamp": frame_stamp(filename) if found else None,
    }


def handle_map_get(handler, path, query, map_template=TEMPLATE_FILE):
    """Serve a map route on any BaseHTTPRequestHandler. Returns True if the path was handled."""
    if path in ("/", "/index.html", "/map"):
        _send(handler, map_template.read_bytes(), "text/html; charset=utf-8")
    elif path == "/api/map/cameras":
        _send_json(handler, cameras_payload())
    elif path == "/api/map/frame":
        cam_id = query.get("camera_id", [""])[0].strip()
        filename = query.get("filename", [""])[0]
        _send_json(handler, frame_payload(cam_id, filename))
    elif path.startswith("/images/"):
        parts = path[len("/images/"):].split("/")
        full = resolve_frame(parts[0], parts[1]) if len(parts) == 2 else None
        if full is None:
            handler.send_error(404, "Image not found")
        else:
            ctype = mimetypes.guess_type(full.name)[0] or "image/jpeg"
            _send(handler, full.read_bytes(), ctype, cache=True)
    else:
        return False
    return True


def _send(handler, body, content_type, status=200, cache=False):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    if cache:
        handler.send_header("Cache-Control", "public, max-age=86400")
    handler.end_headers()
    handler.wfile.write(body)


def _send_json(handler, data, status=200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    _send(handler, body, "application/json; charset=utf-8", status=status)


class MapRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        if not handle_map_get(self, path, urllib.parse.parse_qs(url.query)):
            self.send_error(404, "Not found")

    def log_message(self, fmt, *args):
        pass


def start_server(port=8081):
    load_cameras()
    with_images = sum(1 for c in CAMERAS if c["image_count"])
    server = ThreadingHTTPServer(("", port), MapRequestHandler)
    print("=" * 60)
    print(" Bangkok CCTV Camera Map (map only)")
    print(f" Cameras loaded: {len(CAMERAS)} ({with_images} with images)")
    print(f" Local URL: http://localhost:{port}")
    print(" For the map + Detect together, run: python Apps/main.py")
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
