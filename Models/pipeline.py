import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import cv2

try:
    from .detector import YOLOVehicleDetector, DEFAULT_VEHICLE_CLASSES, imread_unicode, imwrite_unicode
except ImportError:
    from detector import YOLOVehicleDetector, DEFAULT_VEHICLE_CLASSES, imread_unicode, imwrite_unicode
try:
    from .model_registry import resolve_model
except ImportError:
    from model_registry import resolve_model

FILENAME_PATTERN = re.compile(r'^(\d+)_(.+)_(\d{8})_(\d{6})\.jpe?g$', re.IGNORECASE)

# Frames from frozen camera feeds (same picture as the previous frame), made by
# DataEngineering/quality/find_frozen_frames.py. Counting them would repeat old traffic.
DEFAULT_FROZEN_CSV = Path(__file__).resolve().parent.parent / "DataEngineering" / "quality" / "frozen_frames.csv"

def parse_cctv_filename(filename: str) -> Optional[Dict]:
    """
    Parses standard BMA CCTV image filename format:
    e.g. '1103_แยกรัชดา-ราชพฤกษ์-ด้านถนนรัช_รถมาจากแยกท่าพระ_20260830_121547.jpg'
    """
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return None
    cam_id, desc, date_str, time_str = m.groups()
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
    iso_timestamp = f"{formatted_date}T{formatted_time}"
    return {
        "camera_id": cam_id,
        "description": desc,
        "date": formatted_date,
        "time": formatted_time,
        "timestamp": iso_timestamp,
        "hour": int(time_str[:2]),
        "minute": int(time_str[2:4]),
        "second": int(time_str[4:])
    }

def load_cctv_metadata(csv_path: Path) -> Dict[str, Dict]:
    """Loads location, lat, long from cctv_info.csv (cp874 encoded)."""
    meta = {}
    if not csv_path.exists():
        return meta
    try:
        with open(csv_path, mode="r", encoding="cp874", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("camera_id")
                if cid:
                    meta[cid] = row
    except Exception as e:
        print(f"[Warning] Failed loading cctv_info.csv: {e}")
    return meta

def load_frozen_keys(csv_path: Path) -> set:
    """(camera_id, ISO timestamp) of every frozen frame; empty set if the list is missing."""
    if not csv_path or not Path(csv_path).exists():
        return set()
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        return {(int(r["camera_id"]), r["timestamp"]) for r in csv.DictReader(f)}


def drop_frozen(image_paths: List[Path], frozen_keys: set):
    """Returns (paths to keep, number skipped). Files whose name can't be parsed are kept."""
    kept, skipped = [], 0
    for p in image_paths:
        parsed = parse_cctv_filename(p.name)
        if parsed and (int(parsed["camera_id"]), parsed["timestamp"]) in frozen_keys:
            skipped += 1
        else:
            kept.append(p)
    return kept, skipped


def run_batch_pipeline(
    input_dir: Path,
    output_csv: Path,
    cctv_info_csv: Optional[Path] = None,
    annotated_output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    conf_threshold: float = 0.25,
    model_name: str = "yolov8n.pt",
    device: str = "cpu",
    frozen_csv: Optional[Path] = DEFAULT_FROZEN_CSV
):
    """
    Processes all CCTV images in input_dir, extracts vehicle detections,
    and writes tabular dataset ready for Data Science & Analytics.
    """
    detector = YOLOVehicleDetector(
        model_name=model_name,
        confidence_threshold=conf_threshold,
        device=device
    )

    meta_lookup = load_cctv_metadata(cctv_info_csv) if cctv_info_csv else {}

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(input_dir.rglob(ext))

    image_paths = sorted(image_paths)
    image_paths, n_frozen = drop_frozen(image_paths, load_frozen_keys(frozen_csv))
    if n_frozen:
        print(f"[*] Skipped {n_frozen} frozen-camera frames listed in {frozen_csv}")
    if limit:
        image_paths = image_paths[:limit]

    print(f"[*] Found {len(image_paths)} images to process.")

    if annotated_output_dir:
        annotated_output_dir.mkdir(parents=True, exist_ok=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_file", "camera_id", "timestamp", "date", "hour", "minute",
        "latitude", "longitude", "location_en",
        "total_vehicles", "car", "motorcycle", "bus", "truck", "bicycle"
    ]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, img_p in enumerate(image_paths, 1):
            parsed = parse_cctv_filename(img_p.name)
            cam_id = parsed["camera_id"] if parsed else "unknown"
            t_stamp = parsed["timestamp"] if parsed else ""
            d_str = parsed["date"] if parsed else ""
            hour = parsed["hour"] if parsed else ""
            minute = parsed["minute"] if parsed else ""

            cam_info = meta_lookup.get(cam_id, {})
            lat = cam_info.get("latitude", "")
            lon = cam_info.get("longitude", "")
            loc_en = cam_info.get("location_en", "")

            # Predict
            res = detector.predict(img_p, draw_annotations=bool(annotated_output_dir))
            counts = res["counts"]

            row = {
                "image_file": img_p.name,
                "camera_id": cam_id,
                "timestamp": t_stamp,
                "date": d_str,
                "hour": hour,
                "minute": minute,
                "latitude": lat,
                "longitude": lon,
                "location_en": loc_en,
                "total_vehicles": counts.get("total_vehicles", 0),
                "car": counts.get("car", 0),
                "motorcycle": counts.get("motorcycle", 0),
                "bus": counts.get("bus", 0),
                "truck": counts.get("truck", 0),
                "bicycle": counts.get("bicycle", 0)
            }
            writer.writerow(row)

            if annotated_output_dir and res["annotated_image"] is not None:
                out_img_path = annotated_output_dir / f"pred_{img_p.name}"
                imwrite_unicode(out_img_path, res["annotated_image"])

            if idx % 50 == 0 or idx == len(image_paths):
                print(f"[{idx}/{len(image_paths)}] Processed: {img_p.name} -> Total Vehicles: {counts.get('total_vehicles', 0)}")

    print(f"[+] Completed! Results exported to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangkok CCTV YOLO Vehicle Processing Pipeline")
    parser.add_argument("--input-dir", type=str, default="dataset/100_cameras/cctv_images", help="Path to input images directory")
    parser.add_argument("--output-csv", type=str, default="DataEngineering/vehicle_counts.csv", help="Output CSV path")
    parser.add_argument("--cctv-info", type=str, default="dataset/100_cameras/cctv_info.csv", help="Path to cctv_info.csv")
    parser.add_argument("--save-annotated", type=str, default="", help="Directory to save annotated images (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--model", type=str, default=None, help="YOLO weights (default: Models/weights/best.pt if present, else yolov8n.pt)")
    parser.add_argument("--keep-frozen", action="store_true",
                        help="Also count frames from frozen camera feeds (skipped by default)")
    args = parser.parse_args()

    ann_dir = Path(args.save_annotated) if args.save_annotated else None
    run_batch_pipeline(
        input_dir=Path(args.input_dir),
        output_csv=Path(args.output_csv),
        cctv_info_csv=Path(args.cctv_info) if args.cctv_info else None,
        annotated_output_dir=ann_dir,
        limit=args.limit,
        conf_threshold=args.conf,
        model_name=resolve_model(args.model),
        frozen_csv=None if args.keep_frozen else DEFAULT_FROZEN_CSV
    )
