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
  * frames reviewed in the review tool (`DataEngineering/reviews/train/`, see `DataEngineering/labeling/README.md`)
    use the human labels and are always kept, even if re-sampling would skip them; rejected frames are dropped;
  * everything else is **pseudo-labeled** by a large teacher model (`yolo11x.pt`, conf 0.35). The small student then learns Bangkok-camera specifics from the big model.
* Split by **day**: `2026-08-31` = val (`--val-dates`), the rest = train. Frames from the same day/hour look alike, so a per-image random split would overstate val scores.
* Train split only (val is never touched, so val scores stay comparable):
  * **Stretch / squeeze** (`--stretch`, on by default; `--no-stretch` to skip): each train image gets one extra copy
    `<stem>_stretch.jpg`, resized horizontally by a seeded random factor in 0.8–1.2 (height unchanged).
    The label is copied as-is, since YOLO coordinates are normalized. Mimics cameras with different aspect ratios / lenses.
  * **Bus oversampling** (`--oversample-bus 2`; `1` = off): frames whose label contains a bus (class 2) are listed
    N times in `train.txt`. Buses are only ~3 % of frames, so the model sees too few of them otherwise.
  * **Trucks are not up-weighted yet.** Many yolo11x "truck" boxes are pickups (team rule: pickup = car);
    up-weighting them would teach the wrong class. Revisit after the truck frames are human-reviewed.
* Output: `DataEngineering/train_set/` (`images/`, `labels/`, `train.txt`, `dataset.yaml` are git-ignored; `train_manifest.csv` is committed).
  `dataset.yaml` points `train:` at `train.txt` (an image list; Ultralytics keeps duplicate lines) and `val:` at `images/val`.

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
| `shear=2.0`, `perspective=0.0005` | light distortion: camera angle and lens differ a little between sites |
| no rotation (`degrees=0`) | horizons stay level on fixed cameras |

* The run's `best.pt` (highest val mAP50-95) is copied to **`Models/weights/best.pt`** with `best.json` (run, metrics, date). The previous model is kept as `best_prev.pt`. Use `--no-promote` for experiments.

## 3. eval

Scores the pretrained `yolov8n.pt` (the App's previous model, 4 vehicle classes only) and `best.pt` on the same images:

* `DataEngineering/test_set/` once `merge_test_set.py` has run (hand-verified);
* otherwise the val split (pseudo-labels, so only a sanity check).

Two kinds of numbers per model:

* **mAP / precision / recall** — how well the boxes match;
* **count MAE** — average |model count − true count| per frame, per class and total (conf 0.25).
  This is the number that matters for the traffic analysis.

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
