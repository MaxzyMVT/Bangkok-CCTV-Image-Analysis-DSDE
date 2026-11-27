# Computer Vision & Deep Learning Models (`Models/`)

Houses the YOLO11s computer vision inference, model registry, batch counting pipeline, and fine-tuning scripts for low-resolution ($352 \times 288$) Bangkok BMA surveillance footage.

---

## Directory Structure

```text
Models/
├── detector.py                    # YOLOVehicleDetector (IoS containment suppression, agnostic NMS, solar threshold)
├── model_registry.py              # Resolves best.pt -> yolov8n.pt cleanly across apps and scripts
├── pipeline.py                    # Batch inference runner (writes vehicle_counts.csv, skips frozen feeds)
├── test_detection.py              # Standalone single-frame verification script
├── training_prep.py               # Pseudo-labeling & dataset bootstrapping helpers
├── resolution_adaptation_guide.md # Architectural guide for low-res CIF streams
├── training/
│   ├── README.md                  # Build, train & evaluation guide
│   └── train_pipeline.py          # Stratified sampler + YOLO11s fine-tuning & count-MAE evaluator
└── weights/
    ├── best.pt & best.json        # Fine-tuned Bangkok YOLO11s checkpoint & validation metadata
    └── best_prev.pt               # Previous promoted checkpoint (local)
```

---

## Target Vehicle Classes

| Target Class | ID | Scope |
| :--- | :---: | :--- |
| **Car** | `0` | Sedans, pickups, taxis, vans, tuk-tuks, songthaews, minibuses |
| **Motorcycle** | `1` | Motorbikes and scooters (with or without rider) |
| **Bus** | `2` | BMTA city buses and full-size coaches |
| **Truck** | `3` | 6-wheel+ trucks, box trucks, lorries, trailers |

---

## Usage

### 1. Programmatic Detection
```python
from Models.detector import YOLOVehicleDetector, imread_unicode
from Models.model_registry import resolve_model

detector = YOLOVehicleDetector(model_name=resolve_model())
results = detector.detect(imread_unicode("sample_cctv.jpg"))
print(results["counts"])
```

### 2. Batch Extraction
```bash
python Models/pipeline.py --input dataset/100_cameras/cctv_images --output DataEngineering/vehicle_counts.csv
```

### 3. Low-Resolution CIF Engineering
See [`resolution_adaptation_guide.md`](resolution_adaptation_guide.md) and [`training/README.md`](training/README.md) for rectangular inference (`rect=True`), Intersection-over-Smaller (IoS) duplicate box suppression, solar-adaptive confidence thresholding (`0.28` daytime / `0.20` night via `silver.sun_times`), and temporal train/val splits.
