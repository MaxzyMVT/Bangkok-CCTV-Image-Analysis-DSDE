# Test set labeling (YOLO pre-label → keep fully-correct frames)

Goal: a hand-verified **test set** of CCTV frames, built before any train/val split.
YOLO draws the boxes; a person only judges each frame as **correct** or **wrong**.
Only frames where YOLO got *everything* right are kept.

## Pipeline

| Step | Who | Command | Output |
|---|---|---|---|
| 1. Sample + split cameras | once | `python DataEngineering/labeling/sample_test_candidates.py` | `test_set/candidates.csv`, `test_set/camera_assignment.csv` |
| 2. YOLO pre-label | each member (or once) | `python DataEngineering/labeling/prelabel.py --members member_1` | `labeling_work/member_1/` |
| 3. Review | each member | sort previews in File Explorer (below) | `review/correct/`, `review/wrong/` |
| 4. Merge | once, after everyone | `python DataEngineering/labeling/merge_test_set.py` | `test_set/labels/`, `test_manifest.csv`, `review_log.csv`, `dataset.yaml` |

* 100 cameras are split into 3 shares (34/33/33) — see `test_set/camera_assignment.csv`.
* 10 candidates per camera = 2 per `time_period` (Early Morning, Morning Rush, Midday, Evening Rush, Late Evening), spread over different days → 1,000 candidates.
* Pre-labels use `yolo11x.pt` at `conf=0.30`. Use the same model and conf for everyone so the test set is consistent.
* `candidates.csv` is fixed (seed 42). Do **not** re-run step 1 with `--force` after reviewing has started.
* Environment: `conda activate dsde` (has PyTorch CUDA + ultralytics).

## How to review (step 3)

1. Open `labeling_work/<your_name>/review/` in File Explorer, *View → Large icons* (or open the first image and use arrow keys).
2. For each preview, move the file into **`correct/`** or **`wrong/`**. Unsorted files block the merge.
3. Box tags: **C** car (orange), **M** motorcycle (pink), **B** bus (green), **T** truck (blue). The top bar shows counts.

A frame is **correct** only if **all** of these hold:

- Every vehicle you can clearly identify as a vehicle has a box (cars, motorcycles, buses, trucks; including parked ones on the road).
- No box on a non-vehicle (sign, shadow, building, person without motorcycle, reflection).
- Every class is right. Pickup / van / taxi = **car**. Songthaew, tuk-tuk = **car**. Delivery lorry, 6-wheel+ = **truck**. Van-sized minibus = **car**; full-size bus = **bus**.
- Boxes roughly fit the vehicle (no box covering two vehicles, no duplicate box on one vehicle).

Ignore (don't count as a miss): vehicles that are only a few pixels at the far end of the road and that you couldn't identify yourself, vehicles cut off more than ~70 % by the frame edge, and blurred blobs you can't call a vehicle.

When in doubt → **wrong**. Black/frozen/camera-offline frames → **wrong**.

## Cross-check

Pick ~15 previews from another member's `correct/` and check them against the same rules. If you disagree on more than 2, agree on the rules again before merging.

## Known bias

The test set holds only frames that the pretrained YOLO already handles perfectly, so it leans toward easy scenes.
The pretrained model will score close to 100 % on it. Use it for **regression checks** (a fine-tuned model must not get worse here),
and use `review_log.csv` (the share of `wrong` per camera / time period) to report where YOLO struggles.

## Next (not done yet)

* Exclude test frames — and same-camera frames within ±20 min — from the training pool in `Models/training_prep.py`.
* Split train/val by camera or by day, not per image.
