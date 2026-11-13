# Deep Learning Guide: Adapting & Training YOLO for Multi-Resolution CCTV Footage

> **Project:** Bangkok CCTV Image Analysis (CU DSDE 2110403)  
> **Key Dataset Insight:** 99 out of 100 cameras in the BMA dataset are in **CIF resolution (352 × 288 pixels)**, with 1 camera at **348 × 261 pixels**.

---

## 1. The Multi-Resolution & Low-Resolution Challenge

Standard object detection models (e.g. YOLOv8/v11 pretrained on COCO) are optimized for high-resolution images ($640 \times 640$ or $1024 \times 1024$) with balanced object scales. When applied directly to Bangkok CCTV footage, three major challenges arise:

1. **Severe Upscaling Blur & Interpolation Artifacts:**  
   Passing a $352 \times 288$ frame into a default $640 \times 640$ model causes $1.8\times$ bilinear upsampling. This blurs sharp vehicle contours, license plates, and motorcycle silhouettes into ambiguous blobs.
2. **Feature Map Disappearance (Vanishing Small Objects):**  
   In standard YOLO, the smallest detection head (P3) has a stride of 8. For a $352 \times 288$ image, a car at the distant intersection might only be $12 \times 10$ pixels. After downsampling, it shrinks to $\approx 1.5$ pixels on the feature map, making it nearly impossible for convolutional layers to detect.
3. **Perspective Foreshortening & Oblique Angles:**  
   Overpass and pole-mounted cameras view vehicles from elevated angles (top surfaces, roofs), differing sharply from eye-level COCO training samples.
4. **Adverse Environmental Noise:**  
   Wet asphalt reflections (mimicking extra ghost vehicles), nighttime headlight blooming, and low-light sensor noise.

---

## 2. Recommended Engineering Solutions

### Strategy A: Architectural Adaptation & P2 Feature Layer (Stride 4)
Standard YOLO operates with P3 (stride 8), P4 (stride 16), and P5 (stride 32) heads.  
* **Add P2 Detection Layer (Stride 4):**  
  Modifying the model architecture to branch off from the P2 backbone feature map (stride 4) preserves high-frequency spatial details. On a $384 \times 384$ input, P2 provides a $96 \times 96$ resolution grid, enabling detection of distant $8 \times 8$ pixel motorcycles.
* **Architecture YAML:** Create `yolov8-p2.yaml` adding a 4th detection head at stride 4.

---

### Strategy B: Native Aspect-Ratio & Multi-Scale Input (`imgsz`)
* **Stop forcing 640x640 square padding:** Use rectangular inference and training (`rect=True`). A $352 \times 288$ image has an aspect ratio of $1.22:1$. Passing square images forces unnecessary letterbox black bars.
* **Train at Native Multi-Scale:**  
  Set `imgsz=384` with `scale=0.5` during training. This trains the model across random resolutions between $192$ and $480$, making the model robust across low-res analog CCTV ($352 \times 288$) and modern HD streams ($720p$).

---

### Strategy C: Slicing Aided Hyper Inference (SAHI) for Variable Resolutions
If you need a single pipeline that handles both low-resolution ($352 \times 288$) and future high-resolution ($1080p / 4K$) cameras:
* For high-resolution cameras, **do not downsample** the full frame. Downsampling 1080p to 352p destroys information.
* Instead, apply **SAHI (Slicing Aided Hyper Inference)**:
  1. Slice the high-res image into overlapping $320 \times 320$ patches.
  2. Run inference on each slice.
  3. Merge slice bounding boxes with full-frame detections using NMS (Non-Maximum Suppression).
  4. This provides scale-invariance across arbitrary image resolutions.

---

### Strategy D: Domain Augmentation Pipeline (Albumentations)
To train the model to be invariant to CCTV noise without overfitting:
1. **Resolution Jitter / Downscale Augmentation:** Randomly downscale images to $240 \times 180$ and upscale them back to teach the network to identify pixelated cars.
2. **Adverse Lighting & Glare:** Apply `RandomBrightnessContrast`, Gaussian noise, and artificial circular light flares to simulate headlights.
3. **Wet Road & Reflection Suppression:** Random horizontal erasures or vertical color smears to prevent puddle reflections from being scored as real vehicles.

---

### Strategy E: Unsupervised Pseudo-Labeling (Zero-Cost Annotation)
Since the team has **61,319 unannotated CCTV images**:
1. Run a heavy teacher model (e.g. `yolov8x.pt` or `yolo11x.pt`) with Test-Time Augmentation (TTA) on clean daytime frames.
2. Filter for high-confidence detections (`confidence >= 0.70`).
3. Save these predictions as YOLO-formatted `.txt` annotation files.
4. Fine-tune a lightweight model (`yolov8n.pt` or `yolov8s.pt`) using multi-scale training on these pseudo-labels.
5. This yields a domain-adapted model tuned specifically to Bangkok camera vantage points without manual drawing of 10,000 bounding boxes.

---

## 3. Training Command Reference

```bash
# Fine-tuning with multi-scale resolution and rectangular batching
yolo detect train \
    data=DataEngineering/dataset_cctv.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=384 \
    rect=True \
    scale=0.5 \
    mosaic=1.0 \
    mixup=0.15 \
    batch=16 \
    project=runs/train \
    name=bkk_cctv_multires
```

