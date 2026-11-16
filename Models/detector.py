import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

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
        model_name: str = "yolo11s.pt",
        vehicle_classes: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.vehicle_classes = vehicle_classes or DEFAULT_VEHICLE_CLASSES
        # Fine-tuned models have their own class ids (0=car, 1=motorcycle, ...);
        # unless classes were given explicitly, read them from the weights on load.
        self._auto_classes = vehicle_classes is None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy loader for YOLO model to speed up initialization."""
        if self._model is None:
            try:
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
        draw_annotations: bool = True
    ) -> Dict:
        """
        Runs object detection on a single image.
        
        Args:
            image_input: File path (str/Path) or numpy array (BGR).
            conf: Optional confidence threshold override.
            iou: Optional IoU threshold override.
            draw_annotations: If True, draws bounding boxes, labels, and statistics HUD.
            
        Returns:
            Dictionary containing:
                - 'counts': Dict with counts per vehicle class + 'total_vehicles'
                - 'detections': List of detected object dicts [class_name, confidence, bbox]
                - 'annotated_image': np.ndarray (BGR) with annotations if requested
                - 'orig_shape': (height, width)
        """
        conf = conf if conf is not None else self.confidence_threshold
        iou = iou if iou is not None else self.iou_threshold

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
            verbose=False
        )

        detections = []
        counts = {cls_name: 0 for cls_name in self.vehicle_classes.values()}
        counts["total_vehicles"] = 0

        annotated_img = img_bgr.copy() if draw_annotations else None

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coords = [int(v) for v in box.xyxy[0].tolist()] # [x1, y1, x2, y2]
                class_name = self.vehicle_classes.get(cls_id, "vehicle")

                counts[class_name] = counts.get(class_name, 0) + 1
                counts["total_vehicles"] += 1

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": coords
                })

                if draw_annotations:
                    x1, y1, x2, y2 = coords
                    color = CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)

                    # Draw Bounding Box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    # Label tag
                    label = f"{class_name} {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                    
                    # Background pill for text label
                    cv2.rectangle(
                        annotated_img,
                        (x1, max(0, y1 - text_h - baseline - 4)),
                        (x1 + text_w + 4, max(text_h + baseline + 4, y1)),
                        color,
                        -1
                    )
                    cv2.putText(
                        annotated_img,
                        label,
                        (x1 + 2, max(text_h + 2, y1 - 4)),
                        font,
                        font_scale,
                        (0, 0, 0),
                        font_thickness,
                        cv2.LINE_AA
                    )

        if draw_annotations:
            # Build summary string
            total = counts["total_vehicles"]
            hud_text = f"TOTAL: {total} | Cars: {counts.get('car', 0)} | Moto: {counts.get('motorcycle', 0)} | Bus: {counts.get('bus', 0)} | Truck: {counts.get('truck', 0)}"

            # Auto-scale font dynamically to prevent cut-off on lower resolution CCTV images (e.g. 352x288)
            font = cv2.FONT_HERSHEY_SIMPLEX
            margin = max(8, int(w * 0.025))
            avail_w = max(50, w - 2 * margin)
            font_scale = 0.65
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(hud_text, font, font_scale, font_thickness)
            if text_w > avail_w:
                font_scale = font_scale * (avail_w / text_w)
                font_thickness = 1 if font_scale < 0.55 else 2
                (text_w, text_h), _ = cv2.getTextSize(hud_text, font, font_scale, font_thickness)
                if text_w > avail_w:
                    font_scale = max(0.35, font_scale * (avail_w / text_w))
                    (text_w, text_h), _ = cv2.getTextSize(hud_text, font, font_scale, font_thickness)

            # Draw HUD Analytics Bar on top
            hud_h = max(26, int(text_h + 16))
            hud_overlay = annotated_img.copy()
            cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (20, 24, 30), -1)
            cv2.addWeighted(hud_overlay, 0.80, annotated_img, 0.20, 0, annotated_img)

            # Center text horizontally & vertically within the HUD bar
            text_x = max(margin, (w - text_w) // 2)
            text_y = int((hud_h + text_h) / 2) - 1

            cv2.putText(
                annotated_img,
                hud_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
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
