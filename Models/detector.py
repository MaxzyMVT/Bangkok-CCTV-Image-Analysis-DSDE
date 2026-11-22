import os
import cv2
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

try:
    import torch
    import torch._utils
except ImportError:
    pass

# Standard COCO classes for vehicles & traffic entities
DEFAULT_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Distinct colors for drawing detections (BGR format)
CLASS_COLORS = {
    "car": (0, 165, 255),         # Orange
    "motorcycle": (255, 105, 180),# Hot Pink
    "bus": (50, 205, 50),         # Lime Green
    "truck": (255, 191, 0),       # Deep Sky Blue / Amber
    "bicycle": (147, 20, 255),    # Violet
    "person": (230, 216, 173),    # Light Blue / Cyan
}
DEFAULT_BOX_COLOR = (0, 255, 255)  # Yellow fallback

# Day/Night solar lookup table (populated from scraped sun_times_daily.csv or fallback)
SUN_TIMES_CSV = Path(__file__).resolve().parent.parent / "DataEngineering" / "WebScraping" / "raw" / "sun_times" / "sun_times_daily.csv"
SUN_TIMES_TABLE: Dict[str, Tuple[int, int]] = {}


def load_sun_times_lookup() -> Dict[str, Tuple[int, int]]:
    """Loads date -> (sunrise_minute_of_day, sunset_minute_of_day) from lake.silver.sun_times or CSV."""
    global SUN_TIMES_TABLE
    if SUN_TIMES_TABLE:
        return SUN_TIMES_TABLE

    # Prefer pulling directly from PyDuckLake silver layer if available
    try:
        from DataEngineering.lakehouse_pipeline import load_silver_sun_times
        silver_sun = load_silver_sun_times()
        if silver_sun:
            SUN_TIMES_TABLE.update(silver_sun)
            return SUN_TIMES_TABLE
    except Exception:
        pass

    if SUN_TIMES_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(SUN_TIMES_CSV)
            for _, r in df.iterrows():
                d_str = str(r["date"]).strip()
                sr_dt = pd.to_datetime(r["sunrise"])
                ss_dt = pd.to_datetime(r["sunset"])
                sr_min = sr_dt.hour * 60 + sr_dt.minute
                ss_min = ss_dt.hour * 60 + ss_dt.minute
                SUN_TIMES_TABLE[d_str] = (sr_min, ss_min)
        except Exception as e:
            print(f"[detector] Warning loading sun times lookup: {e}")
    return SUN_TIMES_TABLE


def is_daytime(date_str: Optional[str] = None, time_str: Optional[str] = None, hour: Optional[int] = None, minute: Optional[int] = None) -> bool:
    """
    Determines if the given timestamp corresponds to daytime in Bangkok using scraped solar data.
    Default Bangkok solar window: ~06:06 to ~18:30 (sunrise to sunset).
    """
    if hour is not None:
        m = minute if minute is not None else 0
        cur_min = hour * 60 + m
    elif time_str:
        parts = time_str.replace("T", ":").split(":")
        h = int(parts[0]) if len(parts) > 0 else 12
        m = int(parts[1]) if len(parts) > 1 else 0
        cur_min = h * 60 + m
    else:
        return True  # default to daytime if unknown

    lookup = load_sun_times_lookup()
    if date_str and date_str in lookup:
        sr_min, ss_min = lookup[date_str]
        return sr_min <= cur_min <= ss_min

    # Default fallback solar window for Bangkok (~13.7N)
    return (6 * 60 + 6) <= cur_min <= (18 * 60 + 30)


def get_adaptive_confidence(
    date_str: Optional[str] = None,
    time_str: Optional[str] = None,
    hour: Optional[int] = None,
    minute: Optional[int] = None,
    conf_day: float = 0.28,
    conf_night: float = 0.20
) -> float:
    """Returns adaptive confidence threshold based on daytime vs night/headlight conditions."""
    return conf_day if is_daytime(date_str=date_str, time_str=time_str, hour=hour, minute=minute) else conf_night

def imread_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Reads image safely supporting non-ASCII / Thai Unicode filepaths on Windows."""
    try:
        with open(str(path), "rb") as f:
            bytes_data = bytearray(f.read())
            nparr = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_unicode(path: Union[str, Path], img: np.ndarray, ext: str = ".jpg") -> bool:
    """Writes image safely supporting non-ASCII / Thai Unicode filepaths on Windows."""
    try:
        is_success, buffer = cv2.imencode(ext, img)
        if is_success:
            with open(str(path), "wb") as f:
                f.write(buffer)
            return True
    except Exception:
        pass
    return False

class YOLOVehicleDetector:
    """
    Ultralytics YOLO11 Vehicle Detection & Counting Pipeline
    Tailored for Bangkok CCTV road traffic analysis with fine-tuned custom weights.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        vehicle_classes: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        agnostic_nms: bool = True
    ):
        if model_name is None:
            try:
                from .model_registry import resolve_model
                model_name = resolve_model()
            except ImportError:
                from model_registry import resolve_model
                model_name = resolve_model()
        self.model_name = model_name
        self.vehicle_classes = vehicle_classes or DEFAULT_VEHICLE_CLASSES
        # Fine-tuned models have their own class ids (0=car, 1=motorcycle, ...);
        # unless classes were given explicitly, read them from the weights on load.
        self._auto_classes = vehicle_classes is None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.agnostic_nms = agnostic_nms
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def model(self):
        """Lazy loader for YOLO model to speed up initialization (thread-safe)."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        import torch
                        import torch._utils
                        from ultralytics import YOLO
                        print(f"[YOLOVehicleDetector] Loading model weights: {self.model_name}...")
                        self._model = YOLO(self.model_name)
                        names = dict(self._model.names)
                        if self._auto_classes and len(names) != 80:  # not a COCO checkpoint
                            self.vehicle_classes = {int(i): n for i, n in names.items()}
                            print(f"[YOLOVehicleDetector] Custom model classes: {self.vehicle_classes}")
                    except ImportError:
                        raise ImportError(
                            "The 'ultralytics' package is required. "
                            "Please ensure it is installed via: pip install ultralytics"
                        )
        return self._model

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        draw_annotations: bool = True,
        timestamp: Optional[str] = None,
        date_str: Optional[str] = None,
        adaptive_threshold: bool = True,
        agnostic_nms: Optional[bool] = None
    ) -> Dict:
        """
        Runs object detection on a single image.
        
        Args:
            image_input: File path (str/Path) or numpy array (BGR).
            conf: Optional confidence threshold override.
            iou: Optional IoU threshold override.
            draw_annotations: If True, draws bounding boxes, labels, and statistics HUD.
            timestamp: Optional ISO or HH:MM:SS string for adaptive day/night thresholding.
            date_str: Optional YYYY-MM-DD string.
            adaptive_threshold: If True and conf is not explicitly specified, adjusts
                                confidence dynamically based on solar sunrise/sunset data.
            agnostic_nms: If True, suppresses duplicate detections across classes.
            
        Returns:
            Dictionary containing:
                - 'counts': Dict with counts per vehicle class + 'total_vehicles'
                - 'detections': List of detected object dicts [class_name, confidence, bbox]
                - 'annotated_image': np.ndarray (BGR) with annotations if requested
                - 'orig_shape': (height, width)
        """
        if conf is None:
            if adaptive_threshold and (timestamp or date_str or isinstance(image_input, (str, Path))):
                # Try inferring timestamp from filename if path is given
                t_str = timestamp
                d_str = date_str
                if not t_str and isinstance(image_input, (str, Path)):
                    p_name = Path(image_input).name
                    # Standard BMA pattern: e.g. ..._YYYYMMDD_HHMMSS.jpg
                    parts = p_name.replace(".jpg", "").replace(".jpeg", "").split("_")
                    if len(parts) >= 2 and len(parts[-2]) == 8 and len(parts[-1]) == 6 and parts[-2].isdigit() and parts[-1].isdigit():
                        d_str = f"{parts[-2][:4]}-{parts[-2][4:6]}-{parts[-2][6:]}"
                        t_str = f"{parts[-1][:2]}:{parts[-1][2:4]}:{parts[-1][4:]}"
                    conf = get_adaptive_confidence(date_str=d_str, time_str=t_str)
            else:
                conf = self.confidence_threshold
        iou = iou if iou is not None else self.iou_threshold
        use_agnostic = agnostic_nms if agnostic_nms is not None else self.agnostic_nms

        # Load image if file path is provided
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            img_bgr = imread_unicode(img_path)
            if img_bgr is None:
                raise ValueError(f"Could not read image from path: {img_path}")
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
        else:
            raise TypeError("image_input must be a file path or numpy array (BGR).")

        h, w = img_bgr.shape[:2]

        # Run inference filtering by target vehicle class IDs
        model = self.model  # load first: a custom model replaces vehicle_classes
        target_class_ids = list(self.vehicle_classes.keys())
        results = model.predict(
            source=img_bgr,
            conf=conf,
            iou=iou,
            classes=target_class_ids,
            device=self.device,
            verbose=False,
            agnostic_nms=use_agnostic
        )

        detections = []
        counts = {cls_name: 0 for cls_name in self.vehicle_classes.values()}
        counts["total_vehicles"] = 0

        annotated_img = img_bgr.copy() if draw_annotations else None

        if results and len(results) > 0:
            boxes = results[0].boxes
            raw_dets = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coords = [int(v) for v in box.xyxy[0].tolist()] # [x1, y1, x2, y2]
                class_name = self.vehicle_classes.get(cls_id, "vehicle")

                counts[class_name] = counts.get(class_name, 0) + 1
                counts["total_vehicles"] += 1

                raw_dets.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": coords
                })

            detections = sorted(raw_dets, key=lambda d: d["confidence"], reverse=True)

            if draw_annotations:
                # Sort ascending by confidence so higher-confidence labels render on top
                for det in sorted(raw_dets, key=lambda d: d["confidence"]):
                    cls_id = det["class_id"]
                    class_name = det["class_name"]
                    confidence = det["confidence"]
                    coords = det["bbox"]
                    x1, y1, x2, y2 = coords
                    color = CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)

                    # Draw Bounding Box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    # Label tag
                    label = f"{class_name} {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.38
                    font_thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                    # If box is too near top HUD banner, place label inside the box
                    if y1 < 28:
                        lbl_y = min(h - 4, y1 + text_h + 3)
                        box_top = y1
                        box_bot = min(h, y1 + text_h + baseline + 4)
                    else:
                        lbl_y = y1 - 3
                        box_top = max(0, y1 - text_h - baseline - 4)
                        box_bot = y1

                    lbl_x1 = max(0, x1)
                    lbl_x2 = min(w, x1 + text_w + 4)

                    # Background pill for text label
                    cv2.rectangle(
                        annotated_img,
                        (lbl_x1, box_top),
                        (lbl_x2, box_bot),
                        color,
                        -1
                    )
                    cv2.putText(
                        annotated_img,
                        label,
                        (lbl_x1 + 2, lbl_y),
                        font,
                        font_scale,
                        (0, 0, 0),
                        font_thickness,
                        cv2.LINE_AA
                    )

        if draw_annotations:
            # Build sleek, compact summary string
            total = counts["total_vehicles"]
            hud_text = f"TOTAL: {total} | Cars: {counts.get('car', 0)} | Moto: {counts.get('motorcycle', 0)} | Bus: {counts.get('bus', 0)} | Truck: {counts.get('truck', 0)}"

            # Proportionally compact font: highly legible without oversized letters or border clipping
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(hud_text, font, font_scale, font_thickness)

            # Ensure text leaves at least 15% breathing room on margins
            max_allowed_w = int(w * 0.85)
            if text_w > max_allowed_w:
                font_scale = max(0.28, font_scale * (max_allowed_w / text_w))
                (text_w, text_h), baseline = cv2.getTextSize(hud_text, font, font_scale, font_thickness)

            # Compact HUD Analytics Bar on top (sleek 22-24px height)
            hud_h = max(22, int(text_h + 12))
            hud_overlay = annotated_img.copy()
            cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (18, 22, 28), -1)
            cv2.addWeighted(hud_overlay, 0.85, annotated_img, 0.15, 0, annotated_img)

            # Perfectly center text horizontally & vertically within the HUD bar
            text_x = max(6, (w - text_w) // 2)
            text_y = int((hud_h + text_h) / 2) - 1

            cv2.putText(
                annotated_img,
                hud_text,
                (text_x, text_y),
                font,
                font_scale,
                (245, 248, 252),
                font_thickness,
                cv2.LINE_AA
            )

        return {
            "counts": counts,
            "detections": detections,
            "annotated_image": annotated_img,
            "orig_shape": (h, w)
        }

    def train_or_tune(
        self,
        data_yaml: Union[str, Path],
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "runs/detect",
        name: str = "bkk_cctv_yolo",
        device: Optional[str] = None
    ):
        """
        Fine-tunes the YOLO model on a custom annotated dataset.
        
        Args:
            data_yaml: Path to data.yaml formatted for YOLO.
            epochs: Number of training epochs.
            imgsz: Input image resolution.
            batch: Batch size.
            project: Directory to save run results.
            name: Run experiment name.
            device: Training device ('cpu', '0', etc.).
        """
        train_device = device or self.device
        print(f"[YOLOVehicleDetector] Starting fine-tuning with {data_yaml} on {train_device}...")
        return self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device=train_device
        )
