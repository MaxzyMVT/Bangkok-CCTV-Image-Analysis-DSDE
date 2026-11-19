"""
Merge clear double boxes (one vehicle boxed twice, e.g. car + truck on a pickup) in frames that were
already reviewed, using the same rule as the review page (review_store.resolve_overlaps).

Saved labels have no confidence, so each box borrows it from the pre-label box it came from (same class,
IoU >= 0.8). A box a person drew or re-classed has no confidence and is only merged with a same-class
twin; everything doubtful is left for a person (review page: Show -> overlapping boxes).

Usage:
    python DataEngineering/labeling/fix_overlaps.py --set train            # dry run: report only
    python DataEngineering/labeling/fix_overlaps.py --set train --apply    # write the fixes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_store import ReviewStore, _iou, overlap_pairs, resolve_overlaps  # noqa: E402

MATCH_IOU = 0.8
UNKNOWN_CONF = -1.0   # loses every comparison, so a re-classed box never wins against a scored one


def with_conf(labels, pre) -> list[tuple]:
    out = []
    for b in labels:
        scores = [p[5] for p in pre if int(p[0]) == int(b[0]) and _iou(p, b) >= MATCH_IOU]
        out.append((*b[:5], max(scores) if scores else UNKNOWN_CONF))
    return out


def fix_frame(store: ReviewStore, set_name: str, stem: str, apply: bool) -> int:
    """Number of boxes removed from a reviewed frame (written only when apply=True)."""
    labels = store.read_labels(set_name, stem)
    if not labels or not overlap_pairs(labels):
        return 0
    boxes = with_conf(labels, store.read_prelabel(set_name, stem))
    # an unknown score must not count as "the other class scored higher": only same-class twins merge
    scored = [b for b in boxes if b[5] != UNKNOWN_CONF]
    unscored = [b for b in boxes if b[5] == UNKNOWN_CONF]
    kept, _ = resolve_overlaps(scored)
    kept_unscored = []
    for b in unscored:
        twin = any(int(k[0]) == int(b[0]) and _iou(k, b) >= 0.6 for k in kept + kept_unscored)
        if not twin:
            kept_unscored.append(b)
    final = kept + kept_unscored
    dropped = len(boxes) - len(final)
    if dropped and apply:
        order = {tuple(b[:5]): i for i, b in enumerate(labels)}
        final.sort(key=lambda b: order.get(tuple(b[:5]), len(order)))
        store.save(set_name, stem, [tuple(b[:5]) for b in final], "save")
    return dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["train", "test"], required=True)
    ap.add_argument("--apply", action="store_true", help="Write the fixes (default: report only).")
    args = ap.parse_args()

    store = ReviewStore()
    frames = boxes = left = 0
    for stem, status in sorted(store.all_status(args.set, strict=True).items()):
        if status not in ("approved", "corrected"):
            continue
        n = fix_frame(store, args.set, stem, args.apply)
        if n:
            frames += 1
            boxes += n
        if overlap_pairs(store.read_labels(args.set, stem) or []):
            left += 1
    verb = "fixed" if args.apply else "would fix"
    print(f"[{args.set}] {verb} {frames} frames ({boxes} boxes removed); "
          f"{left} frames {'still have' if args.apply else 'have'} overlapping boxes for a person to check")


if __name__ == "__main__":
    main()
