import os
import sys
import json
import base64
import argparse
import threading
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
from Models.model_registry import resolve_model, best_model_info

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
try:
    from DataEngineering.lakehouse_pipeline import load_silver_cameras
    METADATA_LOOKUP = load_silver_cameras()
except Exception:
    METADATA_LOOKUP = load_cctv_metadata(CCTV_INFO_CSV)

# Cache camera image index for fast navigation
CAMERA_INDEX = {}

# Background Lakehouse Pipeline state
PIPELINE_STATE = {
    "running": False,
    "progress": 0,
    "total": 0,
    "percent": 0.0,
    "stage": "idle",  # "idle", "inference", "lakehouse_ingest", "complete", "error"
    "message": "Pipeline idle",
    "last_file": "",
    "error": None,
    "limit": None,
    "stats": None
}
PIPELINE_LOCK = threading.Lock()


def _run_lakehouse_pipeline_worker(limit=None, conf=None, iou=0.45, agnostic_nms=True, adaptive_threshold=True):
    global PIPELINE_STATE
    try:
        with PIPELINE_LOCK:
            PIPELINE_STATE["running"] = True
            PIPELINE_STATE["progress"] = 0
            PIPELINE_STATE["total"] = 0
            PIPELINE_STATE["percent"] = 0.0
            PIPELINE_STATE["stage"] = "inference"
            PIPELINE_STATE["message"] = f"Initializing YOLO inference on {'all frames' if not limit else f'{limit} frames'}..."
            PIPELINE_STATE["last_file"] = ""
            PIPELINE_STATE["error"] = None
            PIPELINE_STATE["limit"] = limit
            PIPELINE_STATE["stats"] = None

        from Models.pipeline import run_batch_pipeline
        import torch

        dev = "0" if torch.cuda.is_available() else "cpu"
        out_csv = ROOT_DIR / "DataEngineering" / "vehicle_counts.csv"

        def on_progress(cur, tot, fname):
            with PIPELINE_LOCK:
                PIPELINE_STATE["progress"] = cur
                PIPELINE_STATE["total"] = tot
                PIPELINE_STATE["percent"] = round((cur / tot) * 100, 1) if tot > 0 else 0.0
                PIPELINE_STATE["last_file"] = fname
                PIPELINE_STATE["message"] = f"YOLO inference: {cur:,}/{tot:,} frames ({PIPELINE_STATE['percent']}%)"

        print(f"[App Pipeline] Starting YOLO batch pipeline (limit={limit}, device={dev}, dynamic_conf={adaptive_threshold and conf is None}, iou={iou}, agnostic_nms={agnostic_nms})...")
        run_batch_pipeline(
            input_dir=DATASET_DIR,
            output_csv=out_csv,
            cctv_info_csv=CCTV_INFO_CSV,
            limit=limit,
            conf_threshold=conf,
            iou_threshold=iou,
            agnostic_nms=agnostic_nms,
            adaptive_threshold=adaptive_threshold,
            device=dev,
            progress_callback=on_progress
        )

        with PIPELINE_LOCK:
            PIPELINE_STATE["stage"] = "lakehouse_ingest"
            PIPELINE_STATE["message"] = "Ingesting predictions into Lakehouse Bronze & Silver layers..."
        print("[App Pipeline] Ingesting into DuckLakehouse Bronze & Silver...")

        from DataEngineering.lakehouse_pipeline import DuckLakehouse
        lake = DuckLakehouse(reset=False)
        lake.initialize_tables()
        bronze_stats = lake.ingest_bronze(force=True)
        silver_stats = lake.transform_silver()
        try:
            lake.catalog.connection.close()
        except Exception:
            pass

        with PIPELINE_LOCK:
            PIPELINE_STATE["running"] = False
            PIPELINE_STATE["stage"] = "complete"
            PIPELINE_STATE["percent"] = 100.0
            PIPELINE_STATE["message"] = "Pipeline completed successfully!"
            PIPELINE_STATE["stats"] = {
                "bronze_ingested": bronze_stats.get("vehicle_counts_raw", 0),
                "silver_transformed": silver_stats.get("vehicle_counts", 0)
            }
        print(f"[App Pipeline] Completed! Bronze: {bronze_stats}, Silver: {silver_stats}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with PIPELINE_LOCK:
            PIPELINE_STATE["running"] = False
            PIPELINE_STATE["stage"] = "error"
            PIPELINE_STATE["error"] = str(e)
            PIPELINE_STATE["message"] = f"Pipeline failed: {str(e)}"


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


_DETECTOR_LOCK = threading.Lock()

def get_detector(conf=0.25, iou=0.45, agnostic_nms=True):
    global DETECTOR
    if DETECTOR is None:
        with _DETECTOR_LOCK:
            if DETECTOR is None:
                print("[App] Initializing YOLO Vehicle Detector...")
                model_path = resolve_model()
                print(f"[App] Using model weights: {model_path}")
                DETECTOR = YOLOVehicleDetector(
                    model_name=model_path,
                    confidence_threshold=conf,
                    iou_threshold=iou,
                    agnostic_nms=agnostic_nms
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
                "location_th": cinfo.get("location_th") or cinfo.get("location", first_parsed["description"] if first_parsed else ""),
                "location_detail_th": cinfo.get("location_detail_th") or cinfo.get("direction_th") or cinfo.get("location_detail", ""),
                "latitude": cinfo.get("latitude", ""),
                "longitude": cinfo.get("longitude", ""),
                "nearest_station_th": cinfo.get("nearest_station_th", ""),
                "nearest_station_en": cinfo.get("nearest_station_en", ""),
                "transit_system": cinfo.get("transit_system", ""),
                "distance_meters": cinfo.get("distance_meters", None),
                "is_near_transit": bool(cinfo.get("is_near_transit", False)),
                "total_images": len(files),
                "image_files": [f.name for f in files]
            }
    CAMERA_INDEX = idx
    print(f"[App] Indexed {len(CAMERA_INDEX)} camera directories.")
    return CAMERA_INDEX

class CCTVAnalysisHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def do_OPTIONS(self):
        try:
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

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
            except OSError as e:
                try:
                    self.send_error(500, f"Error reading image: {str(e)}")
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass
                return

            try:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(img_data)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(img_data)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass
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
                    "location_detail_th": v.get("location_detail_th", ""),
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
            total = 0
            if mode == "time":
                # 16 consecutive 10-minute snapshots of the chosen camera
                if cid not in cam_idx:
                    cid = list(cam_idx.keys())[0] if cam_idx else "1028"
                cam_data = cam_idx.get(cid)
                if not cam_data and cam_idx:
                    cid = list(cam_idx.keys())[0]
                    cam_data = cam_idx[cid]
                total = len(cam_data["image_files"]) if cam_data else 0
                if total > 0:
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
                            "location_en": cam_data["location_en"],
                            "location_th": cam_data.get("location_th", ""),
                            "location_detail_en": cam_data.get("location_detail_en", ""),
                            "location_detail_th": cam_data.get("location_detail_th", "")
                        })
            else:
                # Multi-camera mode: RANDOM snapshots from 16 DISTINCT cameras (from all 100 cameras)
                # Users can shuffle unlimitedly to get fresh random images from random 16 distinct cams
                import random
                cids_with_images = [c for c, cdata in cam_idx.items() if cdata.get("image_files")]
                total = len(cids_with_images)
                sample_count = min(16, total)
                picked_cids = random.sample(cids_with_images, sample_count) if sample_count > 0 else []

                for c_id in picked_cids:
                    cdata = cam_idx[c_id]
                    imgs = cdata["image_files"]
                    pick_f = random.choice(imgs)
                    img_idx = imgs.index(pick_f)
                    parsed = parse_cctv_filename(pick_f)
                    rel_p = f"dataset/100_cameras/cctv_images/{c_id}/{pick_f}"
                    items.append({
                        "index": img_idx,
                        "camera_id": c_id,
                        "filename": pick_f,
                        "path": rel_p,
                        "timestamp": parsed["timestamp"] if parsed else "",
                        "time": parsed["time"] if parsed else "",
                        "date": parsed["date"] if parsed else "",
                        "location_en": cdata["location_en"],
                        "location_th": cdata.get("location_th", ""),
                        "location_detail_en": cdata.get("location_detail_en", ""),
                        "location_detail_th": cdata.get("location_detail_th", "")
                    })

            self._send_json({
                "mode": mode,
                "camera_id": cid,
                "start_index": start_idx,
                "total": total,
                "items": items,
                "count": len(items)
            })
            return

        # 6. Get Model Metadata Info
        elif path == "/api/model-info":
            m_path = resolve_model()
            m_info = best_model_info()
            is_finetuned = "best.pt" in str(m_path)
            self._send_json({
                "model_name": "YOLO11s (Bangkok Fine-Tuned)" if is_finetuned else "YOLO11s (Pretrained)",
                "model_path": str(m_path),
                "is_finetuned": is_finetuned,
                "base_model": m_info.get("base_model", "yolo11s.pt"),
                "val_mAP50": m_info.get("val_mAP50", None),
                "test_mAP50": m_info.get("test", {}).get("mAP50", None),
                "test_count_mae": m_info.get("test", {}).get("count_mae", {}),
                "info": m_info
            })
            return

        # 7. Lakehouse Pipeline Status
        elif path == "/api/pipeline/status":
            with PIPELINE_LOCK:
                state_copy = dict(PIPELINE_STATE)
            self._send_json(state_copy)
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
            conf = float(req_data.get("conf", 0.35))
            iou = float(req_data.get("iou", 0.45))
            ios = float(req_data.get("ios", 0.65))
            agnostic_nms = bool(req_data.get("agnostic_nms", True))
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
                    "location_th": cam_info.get("location", parsed["description"] if parsed else ""),
                    "location_en": cam_info.get("location_en", ""),
                    "location_detail_th": cam_info.get("location_detail", ""),
                    "location_detail_en": cam_info.get("location_detail_en", "")
                }
            else:
                self._send_json({"error": "Either image_path or image_base64 must be provided"}, status=400)
                return

            if img_bgr is None:
                self._send_json({"error": "Failed reading image data"}, status=500)
                return

            try:
                detector = get_detector(conf=conf, iou=iou, agnostic_nms=agnostic_nms)
                pred_result = detector.predict(img_bgr, conf=conf, iou=iou, agnostic_nms=agnostic_nms, ios_thresh=ios, draw_annotations=True)

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
            conf = float(req_data.get("conf", 0.35))
            iou = float(req_data.get("iou", 0.45))
            ios = float(req_data.get("ios", 0.65))
            agnostic_nms = bool(req_data.get("agnostic_nms", True))

            if not image_paths:
                self._send_json({"error": "No image_paths provided"}, status=400)
                return

            results = []
            detector = get_detector(conf=conf, iou=iou, agnostic_nms=agnostic_nms)

            for rel_p in image_paths[:16]:
                f_path = resolve_dataset_image(rel_p)
                if f_path is None:
                    continue
                img = imread_unicode(f_path)
                if img is None:
                    continue
                pred = detector.predict(img, conf=conf, iou=iou, agnostic_nms=agnostic_nms, ios_thresh=ios, draw_annotations=True)
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

        # 3. Trigger Batch Pipeline to Lakehouse
        elif path == "/api/pipeline/start":
            with PIPELINE_LOCK:
                if PIPELINE_STATE["running"]:
                    self._send_json({"error": "Pipeline is already running", "state": PIPELINE_STATE}, status=409)
                    return
            limit = req_data.get("limit")
            if limit is not None:
                try:
                    limit = int(limit)
                    if limit <= 0:
                        limit = None
                except (ValueError, TypeError):
                    limit = None

            conf = req_data.get("conf")
            if conf is not None:
                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    conf = None
            iou = float(req_data.get("iou", 0.45))
            agnostic_nms = bool(req_data.get("agnostic_nms", True))
            adaptive_threshold = bool(req_data.get("adaptive_threshold", True))

            t = threading.Thread(
                target=_run_lakehouse_pipeline_worker,
                kwargs={
                    "limit": limit,
                    "conf": conf,
                    "iou": iou,
                    "agnostic_nms": agnostic_nms,
                    "adaptive_threshold": adaptive_threshold
                },
                daemon=True
            )
            t.start()
            self._send_json({
                "status": "started",
                "limit": limit,
                "conf": conf or "solar-adaptive (0.28 day / 0.20 night)",
                "iou": iou,
                "agnostic_nms": agnostic_nms,
                "message": f"Pipeline started (limit={limit}, dynamic_conf={adaptive_threshold and conf is None})"
            })
            return

        self.send_error(404, "Endpoint not found")

def start_server(port=8080):
    global METADATA_LOOKUP
    # Ensure dataset is downloaded and extracted from instructor's Google Drive if missing
    ensure_dataset()
    if not METADATA_LOOKUP and CCTV_INFO_CSV.exists():
        METADATA_LOOKUP = load_cctv_metadata(CCTV_INFO_CSV)

    build_camera_index()
    active_model = resolve_model()

    # Pre-warm YOLO detector and PyTorch on main thread to avoid multi-thread import race
    try:
        import torch
        import torch._utils
        print("[App] Pre-warming YOLO model on main thread...")
        detector = get_detector()
        _ = detector.model
        print("[App] YOLO model loaded and ready.")
    except Exception as e:
        print(f"[!] Warning pre-warming YOLO detector: {e}")

    server_address = ("", port)
    httpd = HTTPServer(server_address, CCTVAnalysisHandler)
    print(f"\n=================================================================")
    print(f"🚀 Bangkok CCTV YOLO11 Analytics Web Dashboard (Upgraded) Running!")
    print(f"📦 Active Model Weights: {active_model}")
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
