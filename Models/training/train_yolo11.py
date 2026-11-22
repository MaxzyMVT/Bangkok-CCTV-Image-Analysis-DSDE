"""
Direct YOLO11 fine-tuning script with optimal hyperparameters for Bangkok CCTV traffic detection.
Trains on the 100% human-reviewed training dataset with CCTV-specific domain augmentations.

Usage:
    conda activate dsde
    python Models/training/train_yolo11.py
    python Models/training/train_yolo11.py --device cpu   # if running without GPU
    python Models/training/train_yolo11.py --batch 8      # if lower VRAM is needed
"""
import argparse
import os
import sys
from pathlib import Path

# Prevent crash on Windows when both PyTorch and OpenCV / torchvision load OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_YAML = ROOT_DIR / "DataEngineering" / "train_set" / "dataset.yaml"
TEST_YAML = ROOT_DIR / "DataEngineering" / "test_set" / "dataset.yaml"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 on the reviewed Bangkok CCTV dataset.")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Base pretrained checkpoint (e.g., yolo11s.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (default: 25)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=416, help="Image resolution size (default: 416)")
    parser.add_argument("--device", default=0, help="CUDA device index or 'cpu' (default: 0)")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader worker processes (0 is safest on Windows)")
    parser.add_argument("--name", type=str, default="yolo11s_final_human_reviewed", help="Run folder name under runs/train/")
    parser.add_argument("--project", type=str, default=str(ROOT_DIR / "runs" / "train"), help="Project output folder")
    args = parser.parse_args()

    if not TRAIN_YAML.exists():
        sys.exit(f"[x] Error: Training dataset yaml not found at {TRAIN_YAML}. Run train_pipeline build first.")

    print(f"[*] Initializing YOLO fine-tuning with base model: {args.model}")
    print(f"[*] Dataset: {TRAIN_YAML}")
    print(f"[*] Resolution: {args.imgsz}x{args.imgsz} | Batch: {args.batch} | Device: {args.device}")

    # 1. Load pretrained YOLO model
    model = YOLO(args.model)

    # 2. CCTV-Optimized Augmentation & Training Arguments
    train_args = {
        "data": str(TRAIN_YAML),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cos_lr": True,              # Cosine annealing learning rate schedule
        "label_smoothing": 0.05,     # Regularization against borderline vehicle occlusions
        "seed": 42,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "plots": True,

        # CCTV Domain Augmentations (tailored for fixed, low-resolution 352x288 traffic cameras)
        "hsv_h": 0.015,
        "hsv_s": 0.6,                # Saturation jitter for sunny/glare conditions
        "hsv_v": 0.5,                # Brightness jitter for day vs night vs heavy shadows
        "degrees": 0.0,              # Horizons stay strictly level on fixed pole cameras
        "shear": 2.0,                # Slight perspective skew across camera mounting angles
        "perspective": 0.0005,
        "translate": 0.1,
        "scale": 0.5,                # Multi-scale vehicles near and far down the road
        "fliplr": 0.5,               # Symmetric bidirectional traffic flows
        "mosaic": 1.0,
        "close_mosaic": 10,          # Deactivate mosaic for the final 10 epochs for clean convergence
        "mixup": 0.1,
        "erasing": 0.2,              # Occlusion simulation for utility poles, cables, and billboards
    }

    # 3. Start Training
    print("[*] Starting model training...")
    model.train(**train_args)

    # 4. Evaluate against hand-verified Test Set if available
    if TEST_YAML.exists():
        print(f"\n[*] Evaluating best trained checkpoint on hand-verified test set ({TEST_YAML})...")
        try:
            test_metrics = model.val(data=str(TEST_YAML), split="test", imgsz=args.imgsz, device=args.device)
            print(f"[+] Test mAP50:    {test_metrics.box.map50:.4f}")
            print(f"[+] Test mAP50-95: {test_metrics.box.map:.4f}")
        except Exception as e:
            print(f"[!] Test validation notice: {e}")

    print(f"\n[+] Fine-tuning workflow completed. Best weights saved under: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
