# Bangkok CCTV Image Analysis (DSDE Project)

> **Course:** 2110403 Introduction to Data Science and Data Engineering (DSDE-CEDT, Chulalongkorn University)  
> **Core Theme:** Bangkok Traffic Spatial-Temporal Pattern Analysis & Correlation with Environmental/Urban Factors (Rain, Public Transit, Air Quality).

---

## 📁 Repository Structure

```text
Bangkok-CCTV-Image-Analysis-DSDE/
├── Models/
│   ├── detector.py                  # YOLOVehicleDetector class (inference, counting, HUD overlay)
│   ├── pipeline.py                  # Batch feature extraction pipeline (parses CCTV filenames & exports tabular data)
│   ├── test_detection.py            # Quick test script on CCTV footage
│   ├── training_prep.py             # Script to bootstrap domain-adapted YOLO training datasets with pseudo-labels
│   ├── resolution_adaptation_guide.md # In-depth guide on handling various & low resolutions (CIF 352x288)
│   └── __init__.py
├── Apps/
│   ├── main.py                      # ONE server for everything: Camera Map at /, Detect at /inspect (port 8080)
│   ├── Camera-Map/
│   │   ├── app.py                   # Map data + image routes (/api/map/*, /images/*); can also run map-only
│   │   └── templates/index.html     # Leaflet map of all 100 cameras with search and frame previews
│   └── Detection-Classification/
│       ├── app.py                   # Detect REST API (YOLO, timeline, 16-grid, upload)
│       └── templates/index.html     # Detect dashboard (10-min cycler, 16-grid control room)
├── dataset/
│   └── 100_cameras/
│       ├── cctv_info.csv            # Metadata for 100 camera locations (lat, lon, road, intersection)
│       └── cctv_images/             # Raw image captures (61,319 frames across 100 cameras, Aug 25 - 31, 2026)
├── DataEngineering/
│   ├── cctv_images_metadata.csv     # Complete tabular catalog of all 61,319 images (time, date, coords, resolution)
│   ├── generate_metadata_csv.py     # Script to regenerate/update metadata CSV
│   └── vehicle_counts.csv           # Model output vehicle detections table
├── DataAnalysis/                    # Statistical exploratory data analysis & notebooks
└── discord_sync/                    # Charlie bot synced logs and team discussion notes
```

---

## 🚀 Web App: Camera Map + Detect (`Apps/`)

Run everything with one command, from the repository root:
```bash
python Apps/main.py
```

| Page | URL |
|---|---|
| Camera Map | [http://localhost:8080/](http://localhost:8080/) |
| Detect (YOLO, timeline, 16-grid, upload) | [http://localhost:8080/inspect](http://localhost:8080/inspect) |

Use `--port 9000` to pick another port. On first start, a missing dataset is downloaded automatically.

### Moving between CCTV frames and camera locations

**Detect → Map**
* In **Single Inspector**, click the CCTV frame (it shows the label **“ดูตำแหน่งกล้องบนแผนที่”**), or Tab to it and press `Enter`.
* The map zooms to that camera, highlights its pin and previews the **same frame**, with the camera name, direction and capture time.
* In the **16-grid**, clicking a tile still opens it in the Inspector; use the **pin icon** on the tile to open that frame on the map instead.
* Uploaded images have no camera location, so this action is hidden for them. Cameras without coordinates show a notice instead.

**Map → Detect**
* Search by camera ID or name (press `/` to jump to the search box), or click a pin.
* The preview shows the camera's **latest** frame, or the frame you came from if you arrived from Detect.
* Click the preview image or **“เปิดใน Detect”** to open that camera and frame in Detect, with no need to pick it again.
* The map opens on Bangkok; zoom out to see the cameras in the context of all of Thailand.

**Map styles and BTS/MRT**
* The style bar has **Dark / Streets / Minimal / Satellite**. **Minimal** is a light map with less detail: no buildings, POI icons or small-street names, but main roads, rivers, parks and district names stay readable.
* The **“รถไฟฟ้า”** switch shows the BTS/MRT lines in their official colours; it is on by default in Minimal. Station dots appear from zoom 13, and Thai station names from zoom 15. Click a station for its name and lines.
* Camera pins stay on top of the rail layer, and switching styles or the rail layer keeps the view, the open camera and its Replay frame.
* Rail lines and stations come from OpenStreetMap (© OpenStreetMap contributors, ODbL) and are stored in `Apps/Camera-Map/static/bangkok_rail.geojson`. To refresh them:
  ```bash
  python DataEngineering/build_rail_geojson.py
  ```
* Minimal basemap: [OpenFreeMap](https://openfreemap.org) (no API key) © OpenMapTiles, data from OpenStreetMap.

**ภาพย้อนหลัง (Replay) in the map popup**
* Under the preview: ⏮ previous / ▶ play–pause / ⏭ next, a frame slider, and the real capture date and time.
* Pick a day to jump to its first frame. Play steps through the real still frames (~1 s each) and stops at the day's last frame; pressing play there starts the day over.
* These are still frames roughly every 10 minutes, not video. When frames are missing, the popup shows the real time it jumped from and to.
* “เปิดเฟรมนี้ใน Detect” (or clicking the image) opens whichever frame is on screen, and the map URL follows the frame, so refresh returns to it.

**Links, refresh and Back**
* A frame is identified by camera ID + filename in the URL, e.g.
  `/inspect?camera=915&frame=915_TF-KS-01-วงเวียนใหญ่_วงเวียนใหญ่_20260831_210006.jpg`
  (the map uses the same parameters at `/`). Thai filenames work.
* Refresh, Back and pasted links return to the same frame; the Detect URL follows the timeline as you step through frames.
* If a frame can't be found, the page says so first and then shows the camera's latest frame. An unknown camera ID shows a notice too.

### Detect highlights:
1. **10-Minute Snapshot Cycler & Timeline Player:**
   * Step through 10-minute snapshot intervals for **any of the 100 cameras** (~610 images per camera from Aug 25 to Aug 31).
   * **Keyboard Shortcuts:**
     * `←` (Left Arrow / `A`): Previous 10-minute snapshot.
     * `→` (Right Arrow / `D`): Next 10-minute snapshot.
     * `Spacebar`: Play/Pause timelapse auto-slideshow.
     * `R`: Run YOLO detection on the current snapshot.
   * **Auto-Detect on Cycle:** Checkbox to continuously predict vehicle counts as you step through time.
2. **16-Image Control Room Grid View (4×4 Matrix):**
   * View **16 images at once** in a control-room matrix layout.
   * **Two Modes:**
     * *Time Sequence Mode:* 16 consecutive 10-minute snapshots for the same camera (160 minutes of traffic progression).
     * *City Wide Mode:* 16 different Bangkok cameras at the same time slice.
   * **Run YOLO on All 16:** Runs batch deep learning inference across all 16 tiles and displays vehicle count badges simultaneously.
   * Click any tile to zoom directly into the Single Camera Inspector.

---

## 📊 Dataset Catalog (`DataEngineering/cctv_images_metadata.csv`)

Generated by `DataEngineering/generate_metadata_csv.py`, this comprehensive catalog indexes all **61,319 CCTV images** with 27 rich attributes:
* **Identifiers & Paths:** `image_id`, `camera_id`, `filename`, `relative_path`
* **Spatial Data:** `location_th`, `direction_th`, `location_en`, `location_detail_en`, `latitude`, `longitude`, `ip_address`
* **Temporal Breakdown:** `timestamp` (ISO), `date`, `time`, `year`, `month`, `day`, `day_of_week`, `is_weekend`, `hour`, `minute`, `second`, `time_period` (`Early Morning`, `Morning Rush`, `Midday`, `Evening Rush`, `Night`)
* **Camera Specifications:** `width` (352), `height` (288), `aspect_ratio` (1.22), `file_size_bytes`

---

## 🧠 Training Strategy for Multi-Resolution CCTV

Detailed technical documentation is provided in [`Models/resolution_adaptation_guide.md`](Models/resolution_adaptation_guide.md).

* **The Problem:** 99% of BMA cameras are **CIF resolution ($352 \times 288$)**. Upscaling to standard $640 \times 640$ introduces interpolation blur, while distant vehicles ($10-15$ px) disappear on downsampled feature maps.
* **Key Solutions:**
  1. **Native Aspect-Ratio & Multi-Scale Input:** Train at `imgsz=384` with `rect=True` and `scale=0.5` instead of fixed $640 \times 640$.
  2. **P2 High-Resolution Stride 4 Layer:** Add a 4th detection head to retain fine spatial features for small distant cars and motorcycles.
  3. **SAHI (Slicing Aided Hyper Inference):** Patch slicing for mixed resolution cameras (CIF vs 1080p).
  4. **Unsupervised Pseudo-Labeling:** Auto-generate domain-adapted training labels from 61,000 Bangkok frames using `Models/training_prep.py`.
