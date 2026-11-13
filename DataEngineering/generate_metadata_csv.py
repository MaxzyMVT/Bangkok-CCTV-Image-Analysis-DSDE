import os
import re
import csv
import datetime
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset" / "100_cameras" / "cctv_images"
CCTV_INFO_CSV = ROOT_DIR / "dataset" / "100_cameras" / "cctv_info.csv"
OUTPUT_CSV = ROOT_DIR / "DataEngineering" / "cctv_images_metadata.csv"

FILENAME_PATTERN = re.compile(r'^(\d+)_(.+)_(.+)_(\d{8})_(\d{6})\.jpe?g$', re.IGNORECASE)

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def get_time_period(hour: int) -> str:
    if 5 <= hour < 7:
        return "Early Morning"
    elif 7 <= hour < 10:
        return "Morning Rush"
    elif 10 <= hour < 16:
        return "Midday"
    elif 16 <= hour < 20:
        return "Evening Rush"
    elif 20 <= hour < 23:
        return "Late Evening"
    else:
        return "Night"

def load_cctv_info():
    info = {}
    if not CCTV_INFO_CSV.exists():
        return info
    try:
        with open(CCTV_INFO_CSV, mode="r", encoding="cp874", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("camera_id")
                if cid:
                    info[cid] = row
    except Exception as e:
        print(f"[Warning] Failed loading cctv_info.csv: {e}")
    return info

def detect_camera_resolutions():
    """Probes resolution per camera directory once to avoid 60,000+ PIL opens."""
    resolutions = {}
    for cam_dir in DATASET_DIR.iterdir():
        if not cam_dir.is_dir():
            continue
        first_img = next(cam_dir.glob("*.jpg"), None)
        if first_img:
            try:
                with Image.open(first_img) as im:
                    resolutions[cam_dir.name] = (im.width, im.height)
            except Exception:
                resolutions[cam_dir.name] = (352, 288)
        else:
            resolutions[cam_dir.name] = (352, 288)
    return resolutions

def build_metadata_csv():
    print(f"[*] Scanning {DATASET_DIR}...")
    cctv_meta = load_cctv_info()
    cam_resolutions = detect_camera_resolutions()
    print(f"[*] Loaded metadata for {len(cctv_meta)} cameras.")

    fieldnames = [
        "image_id",
        "camera_id",
        "filename",
        "relative_path",
        "location_th",
        "direction_th",
        "location_en",
        "location_detail_en",
        "latitude",
        "longitude",
        "ip_address",
        "timestamp",
        "date",
        "time",
        "year",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "hour",
        "minute",
        "second",
        "time_period",
        "width",
        "height",
        "aspect_ratio",
        "file_size_bytes"
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    total_count = 0
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8-sig") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        cam_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()], key=lambda x: int(x.name) if x.name.isdigit() else x.name)
        for cam_dir in cam_dirs:
            cam_id = cam_dir.name
            c_info = cctv_meta.get(cam_id, {})
            c_w, c_h = cam_resolutions.get(cam_id, (352, 288))
            c_ar = round(c_w / c_h, 2) if c_h > 0 else 0.0

            img_files = sorted(cam_dir.glob("*.jpg"))
            for img_p in img_files:
                total_count += 1
                m = FILENAME_PATTERN.match(img_p.name)
                if m:
                    cid, loc_th, dir_th, date_raw, time_raw = m.groups()
                else:
                    cid = cam_id
                    loc_th, dir_th = "", ""
                    date_raw, time_raw = "20260825", "000000"

                y, mth, d = int(date_raw[:4]), int(date_raw[4:6]), int(date_raw[6:])
                hh, mm, ss = int(time_raw[:2]), int(time_raw[2:4]), int(time_raw[4:])
                dt = datetime.datetime(y, mth, d, hh, mm, ss)
                weekday_name = DAYS_OF_WEEK[dt.weekday()]
                is_wknd = dt.weekday() >= 5
                period = get_time_period(hh)

                rel_path = str(img_p.relative_to(ROOT_DIR)).replace("\\", "/")
                file_sz = img_p.stat().st_size

                row = {
                    "image_id": total_count,
                    "camera_id": cid,
                    "filename": img_p.name,
                    "relative_path": rel_path,
                    "location_th": loc_th,
                    "direction_th": dir_th,
                    "location_en": c_info.get("location_en", ""),
                    "location_detail_en": c_info.get("location_detail_en", ""),
                    "latitude": c_info.get("latitude", ""),
                    "longitude": c_info.get("longitude", ""),
                    "ip_address": c_info.get("ip_address", ""),
                    "timestamp": dt.isoformat(),
                    "date": f"{y:04d}-{mth:02d}-{d:02d}",
                    "time": f"{hh:02d}:{mm:02d}:{ss:02d}",
                    "year": y,
                    "month": mth,
                    "day": d,
                    "day_of_week": weekday_name,
                    "is_weekend": is_wknd,
                    "hour": hh,
                    "minute": mm,
                    "second": ss,
                    "time_period": period,
                    "width": c_w,
                    "height": c_h,
                    "aspect_ratio": c_ar,
                    "file_size_bytes": file_sz
                }
                writer.writerow(row)

    print(f"[+] Successfully generated: {OUTPUT_CSV}")
    print(f"[+] Total images cataloged: {total_count}")

if __name__ == "__main__":
    build_metadata_csv()

