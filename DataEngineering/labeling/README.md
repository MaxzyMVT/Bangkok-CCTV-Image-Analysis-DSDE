# Review tool: approve or fix YOLO boxes

Why: every analysis (rain vs traffic, BTS, bottlenecks) uses **vehicle counts per camera**. YOLO is never exact,
so people check a sample of frames. That gives us (1) a measured counting error for the report and
(2) clean labels to train a better model. Nobody draws from scratch: YOLO (`yolo11x`) already drew the boxes.

| Set | Frames | Used for |
|---|---|---|
| `test` | 1,000 (10 per camera, 2 per time period) | measuring the error — never used for training |
| `train` | 1,200 (12 per camera) | fine-tuning the model |

There is **no per-person split**. Everyone works from the same queue; if several people help, pick different
cameras with the *Camera* filter. Each frame's result is its own small file, so pushes don't conflict.

## Start

```bash
conda activate dsde
git pull
python DataEngineering/labeling/review_app.py
```

Open <http://localhost:8090>. You need `dataset/` (the course images) and nothing else: the YOLO boxes are
already in git under `DataEngineering/reviews/<set>/prelabels/`.

When you stop, commit and push your results:

```bash
git add DataEngineering/reviews && git commit -m "data(reviews): review batch" && git push
```

## Keys

| Key | Action |
|---|---|
| click a box | select it |
| `1` `2` `3` `4` | car / motorcycle / bus / truck — changes the selected box, or the class for new boxes |
| drag on empty image | draw a new box |
| drag inside a box / drag a corner | move / resize |
| `Del` (or `Backspace` / Mac delete with a box selected) | delete the selected box |
| `Enter` | save and go to the next frame (only once the image has loaded) |
| `R` | reject the frame (unusable) and go next |
| `U` / **↶ Undo last decision** | pressed Enter or R by mistake? puts that frame back exactly as before and reopens it (repeat to go further back) |
| click a row in **Recent decisions** | reopen a frame you just decided, to check or change it |
| `Backspace` with nothing selected | go back to YOLO's boxes — on a saved frame it asks first, then **deletes that saved review** |
| `←` `→` | previous / next frame without saving |
| `H` | hide boxes to see the vehicles underneath |
| `Ctrl+Z` | undo |

After `Enter` the frame is **approved** if you changed nothing (YOLO was right) or **corrected** if you did.
Both count as reviewed. Use *Sort → lowest confidence first* to do the doubtful frames first.

## Rules

- Box **every vehicle you can recognise**, including parked ones on the road.
- **car** = sedan, pickup, van, taxi, tuk-tuk, songthaew, minibus. **bus** = full-size bus.
  **truck** = goods vehicle: lorry, 6-wheel+, box truck, trailer. **motorcycle** = motorbike with or without rider.
- No box for: bicycles, people on foot, signs, shadows, reflections, a vehicle too small or blurred to tell what it is,
  a vehicle more than ~70 % outside the frame.
- One box per vehicle, roughly tight. Two vehicles under one box → split it.
- **Reject** frames that are black, frozen, camera-offline or too smeared to count.
- When unsure about the class, pick the closest; when unsure whether it's a vehicle at all, leave it out.

## After reviewing

```bash
python DataEngineering/labeling/merge_test_set.py          # test set + YOLO count error (count_error.csv)
python Models/training/train_pipeline.py all --device 0     # rebuild train set with your labels, train, eval
```

`merge_test_set.py` refuses to run while test frames are still pending (add `--allow-unreviewed` for a preview).
`train_pipeline.py build` always keeps reviewed train frames and drops rejected ones.

## Notes

- Some cameras watch the same road (≈21 camera pairs within 100 m). Review each frame on its own;
  double counting between cameras is handled later when counts are aggregated per road and direction.
- Re-running `prelabel.py` never overwrites a reviewed frame.
