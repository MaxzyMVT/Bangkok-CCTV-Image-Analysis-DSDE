# Interactive Web Applications (`Apps/`)

This directory houses the unified web application suite for the Bangkok CCTV Image Analysis platform, combining geospatial GIS visualization with real-time deep learning detection and multi-camera control room interfaces.

---

## 🚀 Running the Web Suite

All components are served via a unified, zero-external-framework Python web server:

```bash
# Launch both Camera Map (/) and Detection Inspector (/inspect) on port 8080
python Apps/main.py

# Optional: customize port
python Apps/main.py --port 9000
```

### Application Routes

| Path | Interface | Purpose |
| :--- | :--- | :--- |
| `/` | **Camera Map** | Interactive Leaflet GIS map with custom CCTV beacons, transit overlays, and frame timeline replay. |
| `/inspect` | **Vehicle Inspector** | YOLO deep learning inference, vehicle count breakdown, 10-minute snapshot cycler, and 16-grid control room. |

---

## 🗺️ Camera Map Interface (`Apps/Camera-Map/`)

An interactive mapping environment built with **Leaflet.js**, featuring:

- **Custom CCTV Beacons**: High-visibility markers with inline camera icons and dual-contrast outlines (dark perimeter + bright rim) ensuring high legibility over dark, street, and satellite basemaps.
- **Dynamic Basemap Styles**:
  - `Dark`: CartoDB Dark Matter for command-center night visibility.
  - `Streets`: Standard OpenStreetMap road layout with street names and landmarks.
  - `Minimal`: Clean OpenFreeMap vector style highlighting major road arteries without visual clutter.
  - `Satellite`: Esri World Imagery providing real aerial photography of intersections.
- **Mass Transit Overlay (BTS & MRT)**: Real route alignments and station markers in official line colors. Stations display bilingual names and interchange tags.
- **Historical Frame Replay (ภาพย้อนหลัง)**: Integrated preview player inside camera popups with timeline scrubbing, next/previous frame cycling, and automated slideshow playback.

---

## 🚗 Detection & Control Room (`Apps/Detection-Classification/`)

A traffic analytics interface powered by YOLO vehicle detection models:

- **Single Camera Inspector**:
  - Detects and counts **Cars**, **Motorcycles**, **Buses**, and **Trucks**.
  - Renders colored bounding boxes and a telemetric HUD displaying vehicle counts and confidence scores.
  - **Keyboard Controls**:
    - `←` or `A`: Previous 10-minute snapshot.
    - `→` or `D`: Next 10-minute snapshot.
    - `Spacebar`: Toggle playback of timelapse slideshow.
    - `R`: Re-run YOLO detection on current frame.
  - **Auto-Detect Mode**: Automatically performs model inference whenever the timeline advances.
- **16-Camera Control Room (4×4 Matrix)**:
  - **Time Sequence Mode**: 16 consecutive frames for one camera covering 160 continuous minutes of traffic flow.
  - **City-Wide Mode**: 16 different Bangkok intersections sampled at the exact same point in time.
  - **Batch Detection**: Evaluates vehicle counts across all 16 tiles simultaneously.

---

## 🔗 Bi-Directional Deep Linking

The application supports state synchronization between the Map and Detect views via URL search parameters:
- `?camera=<camera_id>&frame=<filename>`

### Workflows:
1. **Map to Detect**: Clicking any camera preview or "Open in Detect" (`เปิดใน Detect`) navigates to `/inspect` with that camera and frame preselected.
2. **Detect to Map**: In the inspector, clicking "View Camera on Map" (`ดูตำแหน่งกล้องบนแผนที่`) navigates to `/`, flies the map camera to that coordinate, highlights the beacon, and loads the frame in the popup player.

---

## 🔌 API Endpoints

The backend exposes lightweight JSON endpoints consumed by the frontends:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/camera-images?camera_id=<id>` | `GET` | Returns list of available captured frames with timestamps for a camera. |
| `/api/cctv-info` | `GET` | Returns spatial coordinates, descriptions, and road names for all cameras. |
| `/api/detect` | `POST` | Executes YOLO inference on a target frame or uploaded image and returns bounding boxes, labels, and counts. |
| `/images/<camera_id>/<filename>` | `GET` | Serves raw CCTV JPEG frames with HTTP caching. |
| `/static/map/<filename>` | `GET` | Serves map assets including `bangkok_rail.geojson`. |
