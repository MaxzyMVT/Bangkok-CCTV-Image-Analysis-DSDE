# Review tool: approve or fix YOLO boxes

Why: every analysis (rain vs traffic, BTS, bottlenecks) uses **vehicle counts per camera**. YOLO is never exact,
so people check a sample of frames. That gives us (1) a measured counting error for the report and
(2) clean labels to train a better model. Nobody draws from scratch: YOLO (`yolo11x`) already drew the boxes.

| Set | Frames | Used for |
|---|---|---|
| `test` | **300 core** (3 per camera, 60 per time period) — must be done; 700 more optional | measuring the error — never used for training |
| `train` | 1,200 (12 per camera) | fine-tuning the model |

**Do the test set first.** It is the only way to measure the counting error. The page shows only the
300 core frames by default (*Frames → core 300*); switch to *all 1,000* only after those are done.
Train frames are optional extra help: unreviewed train frames still get YOLO labels.

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
| mouse wheel / `+` `−` | zoom in and out at the cursor; `0` shows the whole frame again |
| right-drag (or `Space` + drag) | move around while zoomed |
| `Ctrl+Z` | undo |

After `Enter` the frame is **approved** if you changed nothing (YOLO was right) or **corrected** if you did.
Both count as reviewed. Use *Sort → lowest confidence first* to do the doubtful frames first, or
*Sort → bus/truck first* to do the frames whose pre-label has a bus or truck first (rare classes; many "trucks" are pickups = car),
or *Sort → one per camera first* to spread your reviews over all 100 cameras and all times of day.

**Two boxes on one vehicle.** YOLO often leaves a car box *and* a truck box on the same pickup. Boxes that overlap
that much get a **red dashed outline** and a warning; the first `Enter` on such a frame is held back. Delete the
wrong box (click it, `Del`), or press `Enter` again if they really are two vehicles.
*Show → ⚠ overlapping boxes* lists every frame that still has them, reviewed or not.

Clear cases are merged for you before you see the frame (team rule: keep one box, **car**, or **motorcycle**
if one of the two is a motorcycle), but only when YOLO itself scored that class higher than the other one.
On 261 reviewed pairs that was right ~98 % of the time; when YOLO scored the truck/bus higher the rule was right
only ~55 %, so those pairs stay red for you to decide. Already-reviewed frames get the same treatment with
`python DataEngineering/labeling/fix_overlaps.py --set train` (dry run; add `--apply` to write).

**Val frames.** *Show → val (31 Aug) to do* lists the train-set frames from 31 Aug that are still pending. The
trainer scores the model on these after every epoch to pick the best one, so human labels here matter most.

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

`merge_test_set.py` refuses to run while **core** test frames are still pending (add `--allow-unreviewed` for a preview).
Reviewed non-core test frames are included too; `review_log.csv` has a `core` column to filter on.
`train_pipeline.py build` always keeps reviewed train frames and drops rejected ones.

## Notes

- Some cameras watch the same road (≈21 camera pairs within 100 m). Review each frame on its own;
  double counting between cameras is handled later when counts are aggregated per road and direction.
- Re-running `prelabel.py` never overwrites a reviewed frame.
