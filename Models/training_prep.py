import os
import sys
import yaml
import shutil
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Models"))

from Models.detector import YOLOVehicleDetector, imread_unicode, imwrite_unicode

# Target vehicle class mapping for custom training
TARGET_CLASSES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck"
}

COCO_TO_TARGET = {
    2: 0, # car -> 0
    3: 1, # motorcycle -> 1
    5: 2, # bus -> 2
    7: 3  # truck -> 3
}

def generate_pseudo_dataset(
    output_dir: Path = ROOT_DIR / "DataEngineering" / "cctv_yolo_dataset",
    samples_per_cam: int = 5,
    conf_threshold: float = 0.60,
    val_split: float = 0.20
):
    """
    Bootstraps a domain-adapted CCTV dataset using high-confidence pseudo-labeling.
    Prepares train/val splits and dataset.yaml ready for YOLO training.
    """
    dataset_cctv = ROOT_DIR / "dataset" / "100_cameras" / "cctv_images"
    if not dataset_cctv.exists():
        print(f"Dataset path {dataset_cctv} does not exist.")
        return

    print(f"[*] Bootstrapping CCTV training dataset at: {output_dir}")
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for p in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        p.mkdir(parents=True, exist_ok=True)

    detector = YOLOVehicleDetector(model_name="yolov8n.pt", confidence_threshold=conf_threshold)

    cam_dirs = sorted([d for d in dataset_cctv.iterdir() if d.is_dir()])
    total_images = 0

    for cdir in cam_dirs:
        files = sorted(cdir.glob("*.jpg"))
        if not files:
            continue
        # Pick diverse spread of daytime images
        stride = max(1, len(files) // samples_per_cam)
        selected = files[::stride][:samples_per_cam]

        for f in selected:
            img = imread_unicode(f)
            if img is None:
                continue
            h, w = img.shape[:2]
            res = detector.predict(img, conf=conf_threshold, draw_annotations=False)
            
            # Format YOLO txt annotations: <class_id> <x_center> <y_center> <width> <height>
            yolo_lines = []
            for d in res["detections"]:
                coco_cls = d["class_id"]
                if coco_cls not in COCO_TO_TARGET:
                    continue
                target_cls = COCO_TO_TARGET[coco_cls]
                x1, y1, x2, y2 = d["bbox"]

                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                yolo_lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # Assign split
            is_val = random.random() < val_split
            dest_img = val_img_dir / f.name if is_val else train_img_dir / f.name
            dest_lbl = (val_lbl_dir / f.name).with_suffix(".txt") if is_val else (train_lbl_dir / f.name).with_suffix(".txt")

            # Copy image and write label
            shutil.copy2(f, dest_img)
            with open(dest_lbl, "w", encoding="utf-8") as lf:
                lf.write("\n".join(yolo_lines) + "\n")

            total_images += 1

    # Write dataset.yaml
    data_yaml_path = output_dir / "dataset.yaml"
    yaml_content = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": TARGET_CLASSES
    }
    with open(data_yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(yaml_content, yf, default_flow_style=False)

    print(f"[+] Prepared {total_images} pseudo-labeled CCTV frames.")
    print(f"[+] YOLO data configuration written to: {data_yaml_path}")
    print("\nTo train the model with multi-scale resolution optimization, run:")
    print(f"yolo detect train data={data_yaml_path} model=yolov8n.pt epochs=30 imgsz=384 rect=True scale=0.5")

if __name__ == "__main__":
    generate_pseudo_dataset(samples_per_cam=3)
