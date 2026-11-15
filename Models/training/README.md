# Training pipeline

```
build ──> train ──> Models/weights/best.pt ──> App / batch pipeline use it automatically
                 └─> eval (pretrained yolov8n vs fine-tuned, on the hand-verified test set)
```

```bash
conda activate dsde
python Models/training/train_pipeline.py all            # build + train + promote + eval
```

Stages can also run on their own: `build`, `train`, `eval`. See `--help` for all options.

## 1. build: training data

* Samples `--per-camera 12` frames per camera (≈1,200 total; Charlie's target was 500–1,500), spread over the 5 time periods and different days.
* **No test leakage:** skips all 1,000 test candidates *and* every frame from the same camera within ±20 min of one (`--buffer-min`).
* Labels:
  * frames found in `--labeled-dirs <dir> ...` (folders with `labels/<stem>.txt`) use those human-checked labels;
  * everything else is **pseudo-labeled** by a large teacher model (`yolo11x.pt`, conf 0.35). The small student then learns Bangkok-camera specifics from the big model.
* Split by **day**: `2026-08-31` = val (`--val-dates`), the rest = train. Frames from the same day/hour look alike, so a per-image random split would overstate val scores.
* Output: `DataEngineering/train_set/` (`images/`, `labels/`, `dataset.yaml` are git-ignored; `train_manifest.csv` is committed).

## 2. train: fine-tune + augmentation

* Base: `yolo11s.pt` (COCO pretrained) → 4 classes: car, motorcycle, bus, truck.
* `imgsz=416` (frames are 352×288), 80 epochs, early stop after 20 epochs without improvement. Fits a 6 GB GPU at `batch=16`.
* Augmentation (`AUGMENT` in the script):

| Setting | Why |
|---|---|
| `hsv_v=0.5`, `hsv_s=0.6` | night, headlight glare, rain, overcast |
| `scale=0.5`, `translate=0.1` | vehicles near and far |
| `fliplr=0.5` | traffic in both directions |
| `mosaic=1.0`, `close_mosaic=10`, `mixup=0.1` | more small objects per batch, plain frames at the end |
| `erasing=0.2` | occlusion by poles, wires, other vehicles |
| no rotation / shear / perspective | cameras are fixed |

* The run's `best.pt` (highest val mAP50-95) is copied to **`Models/weights/best.pt`** with `best.json` (run, metrics, date). The previous model is kept as `best_prev.pt`. Use `--no-promote` for experiments.

## 3. eval

Scores the pretrained `yolov8n.pt` (the App's previous model, 4 vehicle classes only) and `best.pt` on the same images:

* `DataEngineering/test_set/` once `merge_test_set.py` has run (hand-verified);
* otherwise the val split (pseudo-labels, so only a sanity check).

Report: `runs/eval/report_<split>_<date>.json`.

## Using the model

`Models/model_registry.py` decides which weights to load:
`--model` argument → `CCTV_MODEL` env var → `Models/weights/best.pt` → `yolov8n.pt`.

```bash
python Apps/app.py --port 8080                      # uses best.pt if present
python Models/pipeline.py --output-csv DataEngineering/vehicle_counts.csv
set CCTV_MODEL=yolov8n.pt && python Apps/app.py     # force the old pretrained model
```

`/api/predict` responses include `"model"` so you can see which weights answered.
Weights (`*.pt`) are not committed. Share `Models/weights/best.pt` + `best.json` through the team Drive.
