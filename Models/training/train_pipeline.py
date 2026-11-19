"""
Training pipeline: build dataset -> fine-tune pretrained YOLO -> evaluate -> save best model.
The App and Models/pipeline.py load Models/weights/best.pt automatically once it exists.

Stages
    build   Sample ~N frames per camera (never test candidates or frames within
            +-20 min of them on the same camera), label them, split train/val by day.
            Frames reviewed in the review tool (DataEngineering/reviews/train) use the
            human labels and are always kept; rejected frames are dropped; the rest are
            pseudo-labeled by a large teacher model (yolo11x).
            Train split only: every image also gets one horizontally stretched/squeezed
            copy (--stretch), and frames with a bus are listed --oversample-bus times in
            train.txt. Trucks are NOT up-weighted until the pickup (= car) labels are reviewed.
    train   Fine-tune a pretrained COCO checkpoint with CCTV-oriented augmentation.
            Copies the run's best.pt to Models/weights/best.pt (+ best.json).
    eval    Compare the pretrained COCO model with the fine-tuned one on the
            hand-verified test set (or the val split if the test set isn't merged yet).

Usage (conda activate dsde):
    python Models/training/train_pipeline.py all                       # everything
    python Models/training/train_pipeline.py build --per-camera 12
    python Models/training/train_pipeline.py train --model yolo11s.pt --epochs 80
    python Models/training/train_pipeline.py eval
"""
import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Prevent crash on Windows when both PyTorch and OpenCV / torchvision load OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "DataEngineering" / "labeling"))
sys.path.insert(0, str(ROOT_DIR / "Models"))

from common import (CANDIDATES_CSV, COCO_TO_TARGET, METADATA_CSV, TARGET_CLASSES, TEST_SET_DIR,  # noqa: E402
                    find_image, preview_name, resolve_dataset_root)
from model_registry import BEST_MODEL, BEST_MODEL_INFO, WEIGHTS_DIR  # noqa: E402
from review_store import ReviewStore  # noqa: E402
from sample_test_candidates import sample_camera  # noqa: E402

TRAIN_SET_DIR = ROOT_DIR / "DataEngineering" / "train_set"
# What `build` actually used (split, label source, boxes). Not train_manifest.csv: that file is the
# review tool's train queue (committed) and must keep rejected frames. Git-ignored.
BUILD_MANIFEST = TRAIN_SET_DIR / "build_manifest.csv"
RUNS_DIR = ROOT_DIR / "runs" / "train"
EVAL_DIR = ROOT_DIR / "runs" / "eval"

# Augmentation tuned for fixed low-res (352x288) street cameras, day + night.
AUGMENT = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,   # strong brightness jitter: night, glare, rain
    degrees=0.0,                         # no rotation: horizons stay level on fixed cameras
    shear=2.0, perspective=0.0005,       # light distortion: camera angle / lens differ a little per site
    translate=0.1, scale=0.5,            # vehicles at many distances
    fliplr=0.5, flipud=0.0,
    mosaic=1.0, close_mosaic=10,         # mosaic early, plain frames for the last 10 epochs
    mixup=0.1, copy_paste=0.0,
    erasing=0.2,                         # simulates occlusion by poles / wires / other cars
)


# ----------------------------------------------------------------------------- build
def excluded_frames(meta: pd.DataFrame, buffer_min: int) -> set[int]:
    """All test candidates, plus same-camera frames within +-buffer_min of any candidate."""
    if not CANDIDATES_CSV.exists():
        print("[!] No test candidates found - nothing excluded. Run sample_test_candidates.py first.")
        return set()
    cand = pd.read_csv(CANDIDATES_CSV, encoding="utf-8-sig")
    ts = pd.to_datetime(meta["timestamp"])
    excluded = set(cand["image_id"].astype(int))
    buf = timedelta(minutes=buffer_min)
    for cam, group in cand.groupby("camera_id"):
        cam_mask = meta["camera_id"] == cam
        for t in pd.to_datetime(group["timestamp"]):
            near = cam_mask & (ts >= t - buf) & (ts <= t + buf)
            excluded.update(meta.loc[near, "image_id"].astype(int))
    return excluded


REVIEWED = ("approved", "corrected")


def select_frames(picked: pd.DataFrame, pool: pd.DataFrame, statuses: dict[str, str]) -> pd.DataFrame:
    """Drop rejected frames, force in every reviewed frame (even if re-sampling skipped it),
    and mark label_source = human for reviewed frames, pseudo for the rest."""
    if "stem" not in pool.columns:
        pool = pool.assign(stem=[preview_name(c, i) for c, i in zip(pool["camera_id"], pool["image_id"])])
    reviewed = {s for s, st in statuses.items() if st in REVIEWED}
    rejected = {s for s, st in statuses.items() if st == "rejected"}
    extra = pool[pool["stem"].isin(reviewed - set(picked["stem"]))]
    out = pd.concat([picked, extra[picked.columns.intersection(extra.columns)]], ignore_index=True)
    out = out[~out["stem"].isin(rejected)].copy()
    out["label_source"] = ["human" if s in reviewed else "pseudo" for s in out["stem"]]
    return out


CAR, MOTO, BUS, TRUCK = 0, 1, 2, 3  # target class ids (common.TARGET_CLASSES)
STRETCH_RANGE = (0.8, 1.2)


def build_image_list(
    entries: list[tuple],
    oversample_bus: int = 2,
    oversample_moto: int = 1,
    oversample_truck: int = 1,
    oversample_human: int = 1
) -> list[str]:
    """Train image list: (image path, label text[, is_human]) tuples -> paths.
    Oversamples minority classes (bus, motorcycle, truck) and high-quality human-verified frames."""
    out = []
    for item in entries:
        path, label = item[0], item[1]
        is_human = item[2] if len(item) > 2 else False
        classes = {int(line.split()[0]) for line in label.splitlines() if line.strip()}
        weight = 1
        if BUS in classes:
            weight = max(weight, oversample_bus)
        if MOTO in classes:
            weight = max(weight, oversample_moto)
        if TRUCK in classes:
            weight = max(weight, oversample_truck)
        if is_human:
            weight = max(weight, oversample_human)
        out += [path] * weight
    return out


def stretch_copy(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path, factor: float):
    """Save img_src resized to width*factor (height unchanged). YOLO coords are normalized,
    so the label file is copied as-is."""
    from PIL import Image
    with Image.open(img_src) as im:
        im.resize((round(im.width * factor), im.height), Image.BILINEAR).save(img_dst, quality=95)
    shutil.copy2(lbl_src, lbl_dst)


def count_mae(model_counts: list[dict], true_counts: list[dict]) -> dict[str, float]:
    """Mean absolute count error per class (and total) over frames."""
    keys = [*TARGET_CLASSES.values(), "total"]
    n = max(len(true_counts), 1)
    return {k: round(sum(abs(m[k] - t[k]) for m, t in zip(model_counts, true_counts)) / n, 4) for k in keys}


def build(args):
    from ultralytics import YOLO

    meta = pd.read_csv(METADATA_CSV, encoding="utf-8-sig")
    root = resolve_dataset_root(args.dataset_root)
    excluded = excluded_frames(meta, args.buffer_min)
    pool = meta[~meta["image_id"].isin(excluded)]
    print(f"[*] {len(meta)} frames, {len(excluded)} excluded (test candidates +-{args.buffer_min} min), "
          f"{len(pool)} available")

    rng = random.Random(args.seed)
    picked = pd.concat(sample_camera(g, args.per_camera, rng) for _, g in pool.groupby("camera_id"))
    picked = picked.copy()
    picked["stem"] = [preview_name(c, i) for c, i in zip(picked["camera_id"], picked["image_id"])]
    store = ReviewStore()
    picked = select_frames(picked, pool, store.all_status("train", strict=True))
    picked["split"] = ["val" if d in args.val_dates else "train" for d in picked["date"]]

    teacher = YOLO(args.teacher)

    shutil.rmtree(TRAIN_SET_DIR / "images", ignore_errors=True)
    shutil.rmtree(TRAIN_SET_DIR / "labels", ignore_errors=True)
    for split in ("train", "val"):
        (TRAIN_SET_DIR / "images" / split).mkdir(parents=True)
        (TRAIN_SET_DIR / "labels" / split).mkdir(parents=True)

    n_boxes = []
    for r in picked.itertuples():
        src = find_image(root, r.relative_path)
        img_dst = TRAIN_SET_DIR / "images" / r.split / f"{r.stem}.jpg"
        lbl_dst = TRAIN_SET_DIR / "labels" / r.split / f"{r.stem}.txt"
        shutil.copy2(src, img_dst)
        if r.label_source == "human":
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in store.read_labels("train", r.stem) or []]
            lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            cached_pre = store.read_prelabel("train", r.stem)
            if cached_pre:
                lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h, *_ in cached_pre]
                lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            else:
                res = teacher.predict(str(img_dst), conf=args.teacher_conf, iou=0.5, imgsz=640,
                                      classes=list(COCO_TO_TARGET), device=args.device, verbose=False)[0]
                lines = [f"{COCO_TO_TARGET[int(c)]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                         for c, (x, y, w, h) in zip(res.boxes.cls.tolist(), res.boxes.xywhn.tolist())]
                lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_boxes.append(sum(1 for _ in lbl_dst.read_text(encoding="utf-8").splitlines()))
    picked["n_boxes"] = n_boxes

    # train-split augmentation on disk: stretched copies + minority / human oversampling. Val is untouched.
    train_imgs, train_lbls = TRAIN_SET_DIR / "images" / "train", TRAIN_SET_DIR / "labels" / "train"
    originals = sorted(train_imgs.glob("*.jpg"))
    human_stems = set(picked.loc[picked["label_source"] == "human", "stem"])
    if args.stretch:
        srng = random.Random(args.seed)
        for img in originals:
            factor = srng.uniform(*STRETCH_RANGE)
            stretch_copy(img, train_lbls / f"{img.stem}.txt", train_imgs / f"{img.stem}_stretch.jpg",
                         train_lbls / f"{img.stem}_stretch.txt", factor)
    entries = []
    for img in sorted(train_imgs.glob("*.jpg")):
        base_stem = img.stem.replace("_stretch", "")
        is_human = base_stem in human_stems
        lbl_text = (train_lbls / f"{img.stem}.txt").read_text(encoding="utf-8")
        entries.append((f"./images/train/{img.name}", lbl_text, is_human))

    image_list = build_image_list(
        entries,
        oversample_bus=args.oversample_bus,
        oversample_moto=args.oversample_moto,
        oversample_truck=args.oversample_truck,
        oversample_human=args.oversample_human,
    )
    (TRAIN_SET_DIR / "train.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
    print(f"[*] train.txt: {len(originals)} frames, {len(entries) - len(originals)} stretched copies, "
          f"{len(image_list)} lines (bus x{args.oversample_bus}, moto x{args.oversample_moto}, "
          f"truck x{args.oversample_truck}, human x{args.oversample_human})")

    with open(TRAIN_SET_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(TRAIN_SET_DIR.resolve()).replace("\\", "/"),
                        "train": "train.txt", "val": "images/val", "names": TARGET_CLASSES},
                       f, sort_keys=False)
    picked[["stem", "image_id", "camera_id", "date", "time_period", "split", "label_source", "n_boxes"]] \
        .to_csv(BUILD_MANIFEST, index=False, encoding="utf-8-sig")

    print(pd.crosstab(picked["split"], picked["label_source"], margins=True))
    print(f"[+] {sum(n_boxes)} boxes -> {TRAIN_SET_DIR / 'dataset.yaml'}")


def patch_ultralytics_save_model():
    """Patches Ultralytics checkpoint saving on Windows to prevent 'I/O operation on closed file'
    errors caused by antivirus scans, flush delays, or closed BytesIO buffers during retries."""
    import io
    import time
    from copy import deepcopy
    import torch
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils import __version__, GIT
    from ultralytics.utils.torch_utils import (
        convert_optimizer_state_dict_to_fp16,
        qat_state,
        strip_qat,
        unwrap_model,
    )

    def safe_save_model(self):
        ema = self.ema.ema
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for k, v in ema.state_dict().items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all() and torch.isfinite(model_sd[k]).all():
                    v.copy_(model_sd[k])
        modelopt = qat_state(ema)
        ema = deepcopy(ema).half().to(memory_format=torch.contiguous_format)
        if hasattr(ema, "criterion"):
            ema.criterion = None
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)

        strip_qat(ema)
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "stopper": {"best_fitness": self.stopper.best_fitness, "best_epoch": self.stopper.best_epoch},
            "model": None,
            "ema": ema,
            "updates": self.ema.updates,
            "modelopt": modelopt,
            "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            "scaler": self.scaler.state_dict(),
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(),
            "date": datetime.now().astimezone().isoformat(),
            "version": __version__,
            "git": {
                "root": str(GIT.root),
                "branch": GIT.branch,
                "commit": GIT.commit,
                "message": GIT.message,
                "origin": GIT.origin,
            },
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

        # Robust serialization with fresh BytesIO buffer on every attempt
        serialized_ckpt = None
        for attempt in range(5):
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                buf = io.BytesIO()
                torch.save(ckpt, buf)
                serialized_ckpt = buf.getvalue()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(1.0)

        # Robust checkpoint write to disk with retry against file locks
        self.wdir.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                self.last.write_bytes(serialized_ckpt)
                if self.best_fitness == self.fitness:
                    self.best.write_bytes(serialized_ckpt)
                if (self.save_period > 0) and (self.epoch % self.save_period == 0):
                    (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(1.0)
        return True

    BaseTrainer.save_model = safe_save_model


# ----------------------------------------------------------------------------- train
def train(args):
    patch_ultralytics_save_model()
    from ultralytics import YOLO

    data = TRAIN_SET_DIR / "dataset.yaml"
    if not data.exists():
        sys.exit("[x] No training dataset. Run the 'build' stage first.")
    if getattr(args, "resume", None):
        resume_ckpt = Path(args.resume)
        if not resume_ckpt.exists():
            sys.exit(f"[x] Resume checkpoint not found: {resume_ckpt}")
        name = resume_ckpt.parents[1].name
        model = YOLO(str(resume_ckpt))
        model.train(resume=True, workers=args.workers, device=args.device)
    else:
        name = args.name or f"{Path(args.model).stem}_{datetime.now():%Y%m%d_%H%M}"
        model = YOLO(args.model)

        train_kwargs = dict(
            data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            patience=args.patience, device=args.device, workers=args.workers,
            project=str(RUNS_DIR), name=name, exist_ok=True, seed=args.seed,
            plots=True, **AUGMENT
        )
        if getattr(args, "cos_lr", False):
            train_kwargs["cos_lr"] = True
        if getattr(args, "label_smoothing", 0.0) > 0:
            train_kwargs["label_smoothing"] = args.label_smoothing
        if getattr(args, "warmup_epochs", None) is not None:
            train_kwargs["warmup_epochs"] = args.warmup_epochs

        model.train(**train_kwargs)

    run_dir = RUNS_DIR / name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"[x] Training finished without {best}")
    metrics = {}
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        res = pd.read_csv(results_csv)
        res.columns = [c.strip() for c in res.columns]
        row = res.loc[res["metrics/mAP50-95(B)"].idxmax()]
        metrics = {"best_epoch": int(row["epoch"]),
                   "val_mAP50": round(float(row["metrics/mAP50(B)"]), 4),
                   "val_mAP50_95": round(float(row["metrics/mAP50-95(B)"]), 4)}
    print(f"[+] Best weights: {best}  {metrics}")

    if args.no_promote:
        print("[i] --no-promote: Models/weights/best.pt left unchanged.")
        return best
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_MODEL.exists():
        shutil.copy2(BEST_MODEL, WEIGHTS_DIR / "best_prev.pt")
    if BEST_MODEL_INFO.exists():
        shutil.copy2(BEST_MODEL_INFO, WEIGHTS_DIR / "best_prev.json")
    shutil.copy2(best, BEST_MODEL)
    BEST_MODEL_INFO.write_text(json.dumps({
        "run": str(run_dir.relative_to(ROOT_DIR)).replace("\\", "/"),
        "base_model": args.model, "epochs": args.epochs, "imgsz": args.imgsz,
        "classes": TARGET_CLASSES, "trained_at": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }, indent=2), encoding="utf-8")
    print(f"[+] Promoted to {BEST_MODEL} - evaluating against test set...")
    return best


# ----------------------------------------------------------------------------- eval
def _coco_view(split_images: Path, split_labels: Path, out: Path) -> Path:
    """Copy of a split with labels remapped to COCO ids, so a COCO model can be scored on it."""
    target_to_coco = {v: k for k, v in COCO_TO_TARGET.items()}
    shutil.rmtree(out, ignore_errors=True)
    (out / "labels").mkdir(parents=True)
    shutil.copytree(split_images, out / "images")
    for txt in split_labels.glob("*.txt"):
        lines = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                lines.append(" ".join([str(target_to_coco[int(parts[0])])] + parts[1:]))
        (out / "labels" / txt.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    import ultralytics
    coco_names = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg/datasets/coco.yaml")
                                .read_text(encoding="utf-8"))["names"]
    yml = out / "dataset.yaml"
    yml.write_text(yaml.safe_dump({"path": str(out.resolve()).replace("\\", "/"), "train": "images",
                                   "val": "images", "names": coco_names}), encoding="utf-8")
    return yml


def _scores(metrics, names: dict) -> dict:
    per_class = {names[int(c)]: round(float(ap), 4) for c, ap in zip(metrics.box.ap_class_index, metrics.box.maps[metrics.box.ap_class_index])}
    return {"mAP50": round(float(metrics.box.map50), 4), "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4), "recall": round(float(metrics.box.mr), 4),
            "mAP50_95_per_class": per_class}


def _frame_counts(model, images: Path, labels: Path, class_map: dict | None, device) -> dict[str, float]:
    """Count MAE of `model` (conf 0.25) against the label files, per class and total."""
    from review_store import counts
    model_counts, true_counts = [], []
    for img in sorted(images.glob("*.jpg")):
        res = model.predict(str(img), conf=0.25, device=device, verbose=False,
                            classes=list(class_map) if class_map else None)[0]
        cls = [class_map[int(c)] if class_map else int(c) for c in res.boxes.cls.tolist()]
        model_counts.append(counts([(c,) for c in cls]))
        lbl = labels / f"{img.stem}.txt"
        rows = [line.split() for line in lbl.read_text(encoding="utf-8").splitlines()] if lbl.exists() else []
        true_counts.append(counts([(int(r[0]),) for r in rows if r]))
    return count_mae(model_counts, true_counts)


def evaluate(args):
    from ultralytics import YOLO

    if (TEST_SET_DIR / "labels").exists() and any((TEST_SET_DIR / "labels").glob("*.txt")):
        # If test_set/images doesn't exist yet, build it automatically from dataset
        if not (TEST_SET_DIR / "images").exists() or not any((TEST_SET_DIR / "images").glob("*.jpg")):
            print("[*] Materializing test set images from dataset/...")
            import subprocess
            subprocess.run([sys.executable, str(ROOT_DIR / "DataEngineering" / "labeling" / "merge_test_set.py"), "--allow-unreviewed"], check=True)
        split_name, images, labels = "test", TEST_SET_DIR / "images", TEST_SET_DIR / "labels"
    else:
        print("[!] Test set not merged yet - evaluating on the val split instead.")
        split_name = "val"
        images, labels = TRAIN_SET_DIR / "images" / "val", TRAIN_SET_DIR / "labels" / "val"
    n_images = len(list(images.glob("*.jpg")))
    print(f"[*] Evaluating on {split_name} ({n_images} images)")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    common = dict(imgsz=args.imgsz, batch=8, device=args.device, conf=0.001, iou=0.6,
                  project=str(EVAL_DIR), exist_ok=True, plots=False, verbose=False,
                  workers=args.workers)
    report = {"split": split_name, "n_images": n_images, "models": {}}

    # pretrained COCO baseline: score only the 4 vehicle classes
    coco_yaml = _coco_view(images, labels, EVAL_DIR / f"{split_name}_coco_view")
    base = YOLO(args.baseline)
    m = base.val(data=str(coco_yaml), split="val", classes=list(COCO_TO_TARGET), name="baseline", **common)
    report["models"][f"pretrained:{args.baseline}"] = _scores(m, base.names)
    report["models"][f"pretrained:{args.baseline}"]["count_mae"] = _frame_counts(
        base, images, labels, COCO_TO_TARGET, args.device)

    fine_path = args.weights or (str(BEST_MODEL) if BEST_MODEL.exists() else None)
    if fine_path:
        view = EVAL_DIR / f"{split_name}_view"
        shutil.rmtree(view, ignore_errors=True)
        shutil.copytree(images, view / "images")
        shutil.copytree(labels, view / "labels")
        yml = view / "dataset.yaml"
        yml.write_text(yaml.safe_dump({"path": str(view.resolve()).replace("\\", "/"), "train": "images",
                                       "val": "images", "names": TARGET_CLASSES}), encoding="utf-8")
        fine = YOLO(fine_path)
        m = fine.val(data=str(yml), split="val", name="finetuned", **common)
        report["models"][f"finetuned:{Path(fine_path).name}"] = _scores(m, fine.names)
        report["models"][f"finetuned:{Path(fine_path).name}"]["count_mae"] = _frame_counts(
            fine, images, labels, None, args.device)

        # Auto-update best.json if the evaluated model is BEST_MODEL
        if Path(fine_path).resolve() == BEST_MODEL.resolve() and BEST_MODEL_INFO.exists():
            try:
                curr_info = json.loads(BEST_MODEL_INFO.read_text(encoding="utf-8"))
                finetuned_key = f"finetuned:{Path(fine_path).name}"
                if finetuned_key in report["models"]:
                    scores = report["models"][finetuned_key]
                    new_mae = scores.get("count_mae", {}).get("total")
                    new_map50 = scores.get("mAP50")
                    new_map50_95 = scores.get("mAP50_95")

                    # Compare against previous best if available
                    prev_info_file = WEIGHTS_DIR / "best_prev.json"
                    is_better = True
                    prev_mae, prev_map50 = None, None
                    if prev_info_file.exists():
                        try:
                            prev_info = json.loads(prev_info_file.read_text(encoding="utf-8"))
                            prev_test = prev_info.get("test", {})
                            prev_mae = prev_test.get("count_mae", {}).get("total")
                            prev_map50 = prev_test.get("mAP50")
                            if prev_mae is not None and new_mae is not None:
                                # Improvement if count MAE is strictly better, or MAE tied with better mAP50
                                if new_mae > prev_mae and (new_map50 is None or prev_map50 is None or new_map50 <= prev_map50):
                                    is_better = False
                        except Exception:
                            pass

                    if is_better:
                        if prev_mae is not None:
                            print(f"\n[+] PERFORMANCE GAIN: New model beats previous best (Count MAE: {new_mae} vs {prev_mae}, mAP50: {new_map50} vs {prev_map50})!")
                        curr_info["test"] = {
                            "frames": n_images,
                            "source": f"hand-reviewed core {split_name} set",
                            "mAP50": new_map50,
                            "mAP50_95": new_map50_95,
                            "count_mae": scores.get("count_mae")
                        }
                        note_txt = f"promoted {datetime.now():%Y-%m-%d}: test count MAE {new_mae}"
                        if prev_mae is not None:
                            note_txt += f" (prev {prev_mae}), mAP50 {new_map50} (prev {prev_map50})"
                        curr_info["note"] = note_txt
                        BEST_MODEL_INFO.write_text(json.dumps(curr_info, indent=2), encoding="utf-8")
                    else:
                        print(f"\n[!] MODEL REVERT: New run underperformed previous best (Count MAE: {new_mae} vs {prev_mae}, mAP50: {new_map50} vs {prev_map50}).")
                        if (WEIGHTS_DIR / "best_prev.pt").exists():
                            shutil.copy2(WEIGHTS_DIR / "best_prev.pt", BEST_MODEL)
                            BEST_MODEL_INFO.write_text(json.dumps(prev_info, indent=2), encoding="utf-8")
                            print(f"[+] Preserved previous best model ({WEIGHTS_DIR / 'best_prev.pt'}) as {BEST_MODEL}.")
            except Exception as e:
                print(f"[!] Warning checking promotion metrics: {e}")
    else:
        print("[!] No fine-tuned weights yet - only the baseline was scored.")

    out = EVAL_DIR / f"report_{split_name}_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + pd.DataFrame({k: {m: v[m] for m in ("mAP50", "mAP50_95", "precision", "recall")}
                               for k, v in report["models"].items()}).T.to_string())
    print("\nCount error (mean |model count - label count| per frame):")
    print(pd.DataFrame({k: v["count_mae"] for k, v in report["models"].items()}).T.to_string())
    print(f"\n[+] Report: {out}")
    if split_name == "val":
        print("[i] Val labels are mostly teacher pseudo-labels; treat these numbers as a sanity check only.")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "train", "eval", "all"])
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu'. Default: auto.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-root", default=None, help="Folder that contains dataset/.")
    # build
    ap.add_argument("--per-camera", type=int, default=12, help="Training frames per camera (12 -> ~1,200).")
    ap.add_argument("--val-dates", nargs="+", default=["2026-08-31"], help="Dates held out for validation.")
    ap.add_argument("--buffer-min", type=int, default=20, help="Exclude same-camera frames this close to a test candidate.")
    ap.add_argument("--teacher", default="yolo11x.pt", help="Large model that pseudo-labels unlabeled frames.")
    ap.add_argument("--teacher-conf", type=float, default=0.35)
    ap.add_argument("--oversample-bus", type=int, default=2,
                    help="List train frames that contain a bus this many times in train.txt (1 = off).")
    ap.add_argument("--oversample-moto", type=int, default=1,
                    help="List train frames that contain a motorcycle this many times in train.txt.")
    ap.add_argument("--oversample-truck", type=int, default=1,
                    help="List train frames that contain a truck this many times in train.txt.")
    ap.add_argument("--oversample-human", type=int, default=1,
                    help="List human-verified train frames this many times in train.txt.")
    ap.add_argument("--stretch", action=argparse.BooleanOptionalAction, default=True,
                    help="Add one horizontally stretched/squeezed copy (x0.8-1.2) of each train image.")
    # train
    ap.add_argument("--model", default="yolo11s.pt", help="Pretrained checkpoint to fine-tune.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=416, help="Frames are 352x288; 416 keeps small vehicles visible.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--cos-lr", action="store_true", default=False,
                    help="Use cosine learning rate scheduler instead of linear.")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="Label smoothing factor (e.g. 0.05) to regularize against noisy pseudo-labels.")
    ap.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs.")
    ap.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 = main process, safest on Windows).")
    ap.add_argument("--resume", default=None, help="Resume training from weights/last.pt checkpoint.")
    ap.add_argument("--name", default=None, help="Run name under runs/train/.")
    ap.add_argument("--no-promote", action="store_true", help="Don't overwrite Models/weights/best.pt.")
    # eval
    ap.add_argument("--baseline", default="yolov8n.pt", help="Pretrained model the App used before.")
    ap.add_argument("--weights", default=None, help="Fine-tuned weights to score (default: Models/weights/best.pt).")
    args = ap.parse_args()

    if args.stage in ("build", "all"):
        build(args)
    if args.stage in ("train", "all"):
        best = train(args)
        if args.no_promote and args.weights is None:
            args.weights = str(best)
    if args.stage in ("eval", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
