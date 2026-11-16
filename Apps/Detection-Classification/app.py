import os
import sys
import json
import base64
import argparse
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

# Ensure workspace root and Models are importable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Models"))

from Models.detector import YOLOVehicleDetector, DEFAULT_VEHICLE_CLASSES, imread_unicode, imwrite_unicode
from Models.pipeline import parse_cctv_filename, load_cctv_metadata
from Models.model_registry import resolve_model

# Dataset auto-download from instructor's Google Drive
try:
    from DataEngineering.dataset_downloader import ensure_dataset
except ImportError:
    ensure_dataset = lambda: True

# Paths
DATASET_DIR = ROOT_DIR / "dataset" / "100_cameras" / "cctv_images"
CCTV_INFO_CSV = ROOT_DIR / "dataset" / "100_cameras" / "cctv_info.csv"
TEMPLATE_FILE = Path(__file__).resolve().parent / "templates" / "index.html"

# Global detector instance & metadata
DETECTOR = None
METADATA_LOOKUP = load_cctv_metadata(CCTV_INFO_CSV)

# Cache camera image index for fast navigation
CAMERA_INDEX = {}

def frame_belongs_to_camera(camera_id, filename):
    """True if filename is a dataset frame named for camera_id (e.g. '915_..._20260831_210012.jpg')."""
    if not camera_id or not filename or "/" in filename or "\\" in filename:
        return False
    parsed = parse_cctv_filename(filename)
    return parsed is not None and parsed["camera_id"] == str(camera_id)


def resolve_frame(camera_id, filename):
    """Absolute path of a dataset frame, or None if it is not a real frame of that camera."""
    if not frame_belongs_to_camera(camera_id, filename):
        return None
    try:
        full = (DATASET_DIR / str(camera_id) / filename).resolve()
        full.relative_to(DATASET_DIR.resolve())
    except (ValueError, OSError):
        return None
    return full if full.is_file() else None


def resolve_dataset_image(rel_path):
    """Resolve a repo-relative image path ('dataset/100_cameras/cctv_images/<cid>/<file>').

    Returns the absolute Path only if it stays inside cctv_images, is exactly
    <camera_id>/<filename>, and the filename belongs to that camera. Otherwise None.
    """
    if not rel_path or not isinstance(rel_path, str):
        return None
    try:
        full = (ROOT_DIR / rel_path).resolve()
        rel = full.relative_to(DATASET_DIR.resolve())
    except (ValueError, OSError):
        return None
    if len(rel.parts) != 2:
        return None
    return resolve_frame(rel.parts[0], rel.parts[1])


def get_detector(conf=0.25, iou=0.45):
    global DETECTOR
    if DETECTOR is None:
        print("[App] Initializing YOLO Vehicle Detector...")
        model_path = resolve_model()
        print(f"[App] Using model weights: {model_path}")
        DETECTOR = YOLOVehicleDetector(
            model_name=model_path,
            confidence_threshold=conf,
            iou_threshold=iou
        )
    return DETECTOR

def build_camera_index():
    global CAMERA_INDEX
    if CAMERA_INDEX:
        return CAMERA_INDEX
    
    print("[App] Building camera directory index...")
    idx = {}
    if DATASET_DIR.exists():
        cam_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()],
                          key=lambda x: int(x.name) if x.name.isdigit() else x.name)
        for cdir in cam_dirs:
            cid = cdir.name
            files = sorted(cdir.glob("*.jpg"))
            cinfo = METADATA_LOOKUP.get(cid, {})
            first_parsed = parse_cctv_filename(files[0].name) if files else None
            
            idx[cid] = {
                "camera_id": cid,
                "location_en": cinfo.get("location_en", "BMA CCTV Camera"),
                "location_detail_en": cinfo.get("location_detail_en", ""),
                "location_th": cinfo.get("location", first_parsed["description"] if first_parsed else ""),
                "latitude": cinfo.get("latitude", ""),
                "longitude": cinfo.get("longitude", ""),
                "total_images": len(files),
                "image_files": [f.name for f in files]
            }
    CAMERA_INDEX = idx
    print(f"[App] Indexed {len(CAMERA_INDEX)} camera directories.")
    return CAMERA_INDEX

class CCTVAnalysisHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # 1. Main HTML Dashboard (also at /inspect, its address in the combined Apps/main.py server)
        if path in ("/", "/index.html", "/inspect"):
            if TEMPLATE_FILE.exists():
                content = TEMPLATE_FILE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Template index.html not found")
            return

        # 2. Raw Image Streamer (for fast timeline cycling without Base64 overhead)
        elif path == "/api/image":
            rel_path = query.get("path", [""])[0]
            if not rel_path:
                self.send_error(400, "Missing path parameter")
                return
            full_path = resolve_dataset_image(rel_path)
            if full_path is None:
                self.send_error(404, "Image file not found")
                return
            try:
                with open(full_path, "rb") as f:
                    img_data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(img_data)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(img_data)
            except Exception as e:
                self.send_error(500, f"Error reading image: {str(e)}")
            return

        # 3. List all 100 Cameras
        elif path == "/api/cameras":
            cam_idx = build_camera_index()
            cams_list = [
                {
                    "camera_id": cid,
                    "location_en": v["location_en"],
                    "location_th": v["location_th"],
                    "location_detail_en": v["location_detail_en"],
                    "latitude": v["latitude"],
                    "longitude": v["longitude"],
                    "total_images": v["total_images"]
                }
                for cid, v in cam_idx.items()
            ]
            self._send_json({"cameras": cams_list, "total": len(cams_list)})
            return

        # 4. List all chronological images for a specific camera
        elif path == "/api/camera-images":
            cid = query.get("camera_id", [""])[0]
            cam_idx = build_camera_index()
            if cid not in cam_idx:
                self._send_json({"error": f"Camera ID {cid} not found"}, status=404)
                return
            
            cam_data = cam_idx[cid]
            images_info = []
            for fname in cam_data["image_files"]:
                parsed = parse_cctv_filename(fname)
                rel_p = f"dataset/100_cameras/cctv_images/{cid}/{fname}"
                images_info.append({
                    "filename": fname,
                    "path": rel_p,
                    "timestamp": parsed["timestamp"] if parsed else "",
                    "date": parsed["date"] if parsed else "",
                    "time": parsed["time"] if parsed else "",
                    "hour": parsed["hour"] if parsed else 0,
                    "minute": parsed["minute"] if parsed else 0
                })
            
            self._send_json({
                "camera_id": cid,
                "location_en": cam_data["location_en"],
                "location_th": cam_data["location_th"],
                "total_images": len(images_info),
                "images": images_info
            })
            return

        # 5. Get 16-Image Grid Slice
        elif path == "/api/grid-16":
            mode = query.get("mode", ["time"])[0] # 'time' or 'multicam'
            cid = query.get("camera_id", ["1028"])[0]
            start_idx = int(query.get("start_index", [0])[0])
            cam_idx = build_camera_index()

            items = []
            if mode == "time":
                # 16 consecutive 10-minute snapshots of the chosen camera
                if cid not in cam_idx:
                    cid = list(cam_idx.keys())[0]
                cam_data = cam_idx[cid]
                total = len(cam_data["image_files"])
                start_idx = max(0, min(start_idx, total - 1))
                end_idx = min(start_idx + 16, total)
                slice_files = cam_data["image_files"][start_idx:end_idx]

                for f_idx, fname in enumerate(slice_files):
                    parsed = parse_cctv_filename(fname)
                    rel_p = f"dataset/100_cameras/cctv_images/{cid}/{fname}"
                    items.append({
                        "index": start_idx + f_idx,
                        "camera_id": cid,
                        "filename": fname,
                        "path": rel_p,
                        "timestamp": parsed["timestamp"] if parsed else "",
                        "time": parsed["time"] if parsed else "",
                        "date": parsed["date"] if parsed else "",
                        "location_en": cam_data["location_en"]
                    })
            else:
                # Multi-camera mode: 16 different cameras at approximately the same index
                cids = list(cam_idx.keys())
                start_cam = max(0, min(start_idx, len(cids) - 1))
                picked_cids = cids[start_cam:start_cam + 16]

                for c_id in picked_cids:
                    cdata = cam_idx[c_id]
                    if cdata["image_files"]:
                        # Pick image near start_idx % length
                        pick_f = cdata["image_files"][0]
                        parsed = parse_cctv_filename(pick_f)
                        rel_p = f"dataset/100_cameras/cctv_images/{c_id}/{pick_f}"
                        items.append({
                            "index": 0,
                            "camera_id": c_id,
                            "filename": pick_f,
                            "path": rel_p,
                            "timestamp": parsed["timestamp"] if parsed else "",
                            "time": parsed["time"] if parsed else "",
                            "date": parsed["date"] if parsed else "",
                            "location_en": cdata["location_en"]
                        })

            self._send_json({
                "mode": mode,
                "camera_id": cid,
                "start_index": start_idx,
                "items": items,
                "count": len(items)
            })
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_json({"error": "Empty request body"}, status=400)
            return

        try:
            raw_body = self.rfile.read(content_len).decode("utf-8")
            req_data = json.loads(raw_body)
        except Exception as e:
            self._send_json({"error": f"Invalid JSON payload: {str(e)}"}, status=400)
            return

        # 1. Single Image Prediction
        if path == "/api/predict":
            img_path_rel = req_data.get("image_path")
            img_b64 = req_data.get("image_base64")
            conf = float(req_data.get("conf", 0.25))
            iou = float(req_data.get("iou", 0.45))
            classes_filter = req_data.get("classes", {})

            img_bgr = None
            metadata = {}

            if img_b64:
                try:
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(img_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    metadata = {"camera_id": "Uploaded", "timestamp": "Custom"}
                except Exception as e:
                    self._send_json({"error": f"Failed decoding image: {str(e)}"}, status=400)
                    return
            elif img_path_rel:
                full_path = resolve_dataset_image(img_path_rel)
                if full_path is None:
                    self._send_json({"error": f"Image not found: {img_path_rel}"}, status=404)
                    return
                img_bgr = imread_unicode(full_path)
                parsed = parse_cctv_filename(full_path.name)
                cam_id = parsed["camera_id"] if parsed else "Unknown"
                cam_info = METADATA_LOOKUP.get(cam_id, {})
                metadata = {
                    "camera_id": cam_id,
                    "filename": full_path.name,
                    "timestamp": parsed["timestamp"] if parsed else "",
                    "time": parsed["time"] if parsed else "",
                    "date": parsed["date"] if parsed else "",
                    "description": parsed["description"] if parsed else "",
                    "location_en": cam_info.get("location_en", ""),
                    "location_detail_en": cam_info.get("location_detail_en", "")
                }
            else:
                self._send_json({"error": "Either image_path or image_base64 must be provided"}, status=400)
                return

            if img_bgr is None:
                self._send_json({"error": "Failed reading image data"}, status=500)
                return

            try:
                detector = get_detector(conf=conf, iou=iou)
                pred_result = detector.predict(img_bgr, conf=conf, iou=iou, draw_annotations=True)

                filtered_detections = []
                counts = {"total_vehicles": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}

                for d in pred_result["detections"]:
                    cname = d["class_name"]
                    if classes_filter and not classes_filter.get(cname, True):
                        continue
                    filtered_detections.append(d)
                    counts[cname] = counts.get(cname, 0) + 1
                    counts["total_vehicles"] += 1

                _, enc_ann = cv2.imencode(".jpg", pred_result["annotated_image"])
                ann_b64 = base64.b64encode(enc_ann).decode("utf-8")

                _, enc_orig = cv2.imencode(".jpg", img_bgr)
                orig_b64 = base64.b64encode(enc_orig).decode("utf-8")

                self._send_json({
                    "metadata": metadata,
                    "model": Path(detector.model_name).name,
                    "counts": counts,
                    "detections": filtered_detections,
                    "annotated_image_base64": ann_b64,
                    "original_image_base64": orig_b64
                })
            except Exception as e:
                self._send_json({"error": f"Inference error: {str(e)}"}, status=500)
            return

        # 2. Batch Predict 16 Grid
        elif path == "/api/predict-grid":
            image_paths = req_data.get("image_paths", [])
            conf = float(req_data.get("conf", 0.25))
            iou = float(req_data.get("iou", 0.45))

            if not image_paths:
                self._send_json({"error": "No image_paths provided"}, status=400)
                return

            results = []
            detector = get_detector(conf=conf, iou=iou)

            for rel_p in image_paths[:16]:
                f_path = resolve_dataset_image(rel_p)
                if f_path is None:
                    continue
                img = imread_unicode(f_path)
                if img is None:
                    continue
                pred = detector.predict(img, conf=conf, iou=iou, draw_annotations=True)
                _, enc_thumb = cv2.imencode(".jpg", pred["annotated_image"])
                thumb_b64 = base64.b64encode(enc_thumb).decode("utf-8")
                results.append({
                    "path": rel_p,
                    "filename": f_path.name,
                    "counts": pred["counts"],
                    "annotated_thumb_b64": thumb_b64
                })

            self._send_json({"results": results, "total_processed": len(results)})
            return

        self.send_error(404, "Endpoint not found")

def start_server(port=8080):
    global METADATA_LOOKUP
    # Ensure dataset is downloaded and extracted from instructor's Google Drive if missing
    ensure_dataset()
    if not METADATA_LOOKUP and CCTV_INFO_CSV.exists():
        METADATA_LOOKUP = load_cctv_metadata(CCTV_INFO_CSV)

    build_camera_index()
    server_address = ("", port)
    httpd = HTTPServer(server_address, CCTVAnalysisHandler)
    print(f"\n=================================================================")
    print(f"🚀 Bangkok CCTV YOLO Analytics Web Dashboard (Upgraded) Running!")
    print(f"👉 Local URL: http://localhost:{port}")
    print(f"=================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Bangkok CCTV YOLO Web Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    args = parser.parse_args()
    start_server(port=args.port)
