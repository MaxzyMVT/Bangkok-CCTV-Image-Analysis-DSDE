# Computer Vision & Deep Learning Models (`Models/`)

This directory houses the computer vision architecture and deep learning inference pipelines designed to detect, classify, and count urban vehicles from Bangkok BMA traffic surveillance cameras.

---

## 📁 Directory Structure

```text
Models/
├── detector.py                  # YOLOVehicleDetector class (inference, telemetry & overlays)
├── pipeline.py                  # Batch feature extraction pipeline (processes image collections)
├── test_detection.py            # Quick standalone inference test script
├── training_prep.py             # Domain-adaptation dataset preparation & pseudo-labeling
├── resolution_adaptation_guide.md # Technical guide for low-resolution CIF (352x288) streams
└── weights/                     # Pre-trained and domain-fine-tuned YOLO weights
```

---

## 🚘 Vehicle Detection Architecture

The core detector (`detector.py`) uses Ultralytics YOLO architectures optimized for traffic monitoring:

### Target Vehicle Classes
Standard COCO vehicle classes are mapped to four primary urban transportation categories:

| Target Class | ID | COCO Class Origin | Description |
| :--- | :---: | :--- | :--- |
| **Car** | `0` | `car` | Personal passenger automobiles, sedans, SUVs, and taxis. |
| **Motorcycle** | `1` | `motorcycle` | Two-wheeled motorized vehicles, motorcycle taxis, and delivery scooters. |
| **Bus** | `2` | `bus` | BMTA public transit buses, minibuses, and private coach buses. |
| **Truck** | `3` | `truck` | Light commercial trucks, pickup trucks, and heavy cargo vehicles. |

---

## 🚀 Usage & API

### 1. Programmatic Detection (`YOLOVehicleDetector`)
```python
from Models.detector import YOLOVehicleDetector
import cv2

# Initialize detector (automatically loads weights)
detector = YOLOVehicleDetector(model_name="yolov8n.pt", conf_thresh=0.25)

# Perform inference on an image
image = cv2.imread("sample_cctv.jpg")
results = detector.detect(image)

# Access detection counts
print("Vehicle Counts:", results["counts"])
# Output: {'car': 12, 'motorcycle': 18, 'bus': 2, 'truck': 1, 'total': 33}

# Annotated image with bounding boxes & HUD telemetry
annotated_img = results["annotated_image"]
```

### 2. Standalone Verification (`test_detection.py`)
Run inference on a single image to verify model behavior and visualize output annotations:
```bash
python Models/test_detection.py
```

### 3. Batch Extraction Pipeline (`pipeline.py`)
Processes complete folders of CCTV footage, generating structured CSV files containing vehicle counts indexed by timestamp and camera ID:
```bash
python Models/pipeline.py --input dataset/100_cameras/cctv_images --output DataEngineering/vehicle_counts.csv
```

---

## 🔍 Low-Resolution CIF Adaptation

Over 99% of municipal traffic cameras in Bangkok stream at **CIF resolution ($352 \times 288$)**. Standard models trained on high-resolution $640 \times 640$ imagery encounter significant accuracy drop due to interpolation artifacts and feature map downsampling.

Key adaptation techniques implemented in this project (detailed in [`resolution_adaptation_guide.md`](resolution_adaptation_guide.md)):
1. **Aspect-Ratio Preserved Rectangular Inference**: Eliminating non-uniform square stretching by preserving the native $1.22$ aspect ratio (`rect=True`, `imgsz=384`).
2. **P2 High-Resolution Stride-4 Detection Head**: Retaining shallow spatial feature representations so distant 10–15 pixel vehicles remain detectable.
3. **Domain-Specific Pseudo-Labeling (`training_prep.py`)**: Bootstrapping training datasets directly from unannotated Bangkok frames to fine-tune network backbones on local lighting, camera angles, and road geometry.
