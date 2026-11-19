import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "labeling"))

from review_store import ReviewStore, clean_boxes, counts, overlap_pairs, resolve_overlaps  # noqa: E402

STEM = "cam0001_img00001"


class OverlapTest(unittest.TestCase):
    def test_same_vehicle_two_classes_is_flagged(self):
        # YOLO's per-class NMS leaves a car box and a truck box on one pickup
        self.assertEqual(overlap_pairs([(0, .5, .5, .2, .2), (3, .505, .5, .21, .2)]), [(0, 1)])

    def test_neighbouring_vehicles_are_not_flagged(self):
        # a motorcycle beside a car, partly overlapping, is two real vehicles
        self.assertEqual(overlap_pairs([(0, .5, .5, .2, .2), (1, .62, .55, .06, .12)]), [])
        self.assertEqual(overlap_pairs([]), [])


class ResolveOverlapsTest(unittest.TestCase):
    """Team rule for one vehicle boxed twice: keep one box, car (motorcycle if one box is a motorcycle).
    Applied automatically only when YOLO itself scored that class higher (~98 % right on reviewed frames);
    otherwise both boxes stay for a person to decide."""
    CAR, MOTO, BUS, TRUCK = 0, 1, 2, 3

    def test_car_more_confident_than_truck_is_merged_to_car(self):
        boxes, n = resolve_overlaps([(self.CAR, .5, .5, .2, .2, .80), (self.TRUCK, .505, .5, .21, .2, .50)])
        self.assertEqual(n, 1)
        self.assertEqual([b[0] for b in boxes], [self.CAR])

    def test_truck_more_confident_is_left_for_a_person(self):
        pair = [(self.CAR, .5, .5, .2, .2, .40), (self.TRUCK, .505, .5, .21, .2, .70)]
        self.assertEqual(resolve_overlaps(pair), (pair, 0))

    def test_motorcycle_wins_over_car(self):
        boxes, n = resolve_overlaps([(self.CAR, .5, .5, .1, .2, .50), (self.MOTO, .5, .505, .1, .21, .60)])
        self.assertEqual((n, [b[0] for b in boxes]), (1, [self.MOTO]))

    def test_bus_and_truck_without_car_is_left_alone(self):
        pair = [(self.BUS, .5, .5, .3, .3, .90), (self.TRUCK, .5, .5, .31, .3, .40)]
        self.assertEqual(resolve_overlaps(pair), (pair, 0))

    def test_same_class_twice_keeps_the_more_confident_box(self):
        boxes, n = resolve_overlaps([(self.CAR, .5, .5, .2, .2, .40), (self.CAR, .505, .5, .21, .2, .90)])
        self.assertEqual((n, len(boxes)), (1, 1))
        self.assertAlmostEqual(boxes[0][5], .90)

    def test_separate_vehicles_untouched(self):
        boxes = [(self.CAR, .2, .2, .1, .1, .9), (self.TRUCK, .7, .7, .2, .2, .9)]
        self.assertEqual(resolve_overlaps(boxes), (boxes, 0))


class ReviewStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ReviewStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _pre(self, boxes):
        self.store.write_prelabel("test", STEM, boxes)

    def test_pending_frame_shows_merged_prelabel(self):
        # car scored higher than the truck on the same spot: the page opens with one car box
        self._pre([(0, .5, .5, .2, .2, .8), (3, .505, .5, .21, .2, .5), (1, .1, .1, .05, .1, .7)])
        self.assertEqual(sorted(b[0] for b in self.store.current_boxes("test", STEM)), [0, 1])
        self.assertEqual(len(self.store.read_prelabel("test", STEM)), 3)   # the pre-label file is untouched

    def test_save_unchanged_is_approved(self):
        self._pre([(0, .5, .5, .1, .1, .9)])
        self.assertEqual(self.store.save("test", STEM, [(0, .5, .5, .1, .1)], "save")["status"], "approved")

    def test_save_changed_class_is_corrected(self):
        self._pre([(0, .5, .5, .1, .1, .9)])
        self.assertEqual(self.store.save("test", STEM, [(1, .5, .5, .1, .1)], "save")["status"], "corrected")

    def test_small_shift_still_approved(self):
        self._pre([(0, .5, .5, .1, .1, .9)])
        self.assertEqual(self.store.save("test", STEM, [(0, .501, .5, .1, .1)], "save")["status"], "approved")

    def test_extra_box_is_corrected(self):
        self._pre([(0, .5, .5, .1, .1, .9)])
        boxes = [(0, .5, .5, .1, .1), (1, .2, .2, .05, .1)]
        self.assertEqual(self.store.save("test", STEM, boxes, "save")["status"], "corrected")

    def test_empty_frame_approved(self):
        self._pre([])
        self.assertEqual(self.store.save("test", STEM, [], "save")["status"], "approved")
        label = self.store.set_dir("test") / "labels" / f"{STEM}.txt"
        self.assertTrue(label.exists())
        self.assertEqual(label.read_text(encoding="utf-8").strip(), "")
        self.assertEqual(counts([])["total"], 0)

    def test_reject_removes_labels(self):
        self._pre([(0, .5, .5, .1, .1, .9)])
        self.store.save("test", STEM, [(0, .5, .5, .1, .1)], "save")
        self.assertEqual(self.store.save("test", STEM, [], "reject")["status"], "rejected")
        self.assertIsNone(self.store.read_labels("test", STEM))

    def test_reset_back_to_pending(self):
        self._pre([(0, .5, .5, .1, .1, .9)])
        self.store.save("test", STEM, [(1, .5, .5, .1, .1)], "save")
        self.store.save("test", STEM, [], "reset")
        self.assertEqual(self.store.read_status("test", STEM)["status"], "pending")
        self.assertEqual(self.store.current_boxes("test", STEM), [(0, .5, .5, .1, .1)])

    def test_clean_boxes_clamps_and_drops(self):
        out = clean_boxes([(0, 1.0, .5, .2, .2), (1, .5, .5, .001, .2)])
        self.assertEqual(len(out), 1)
        cls, xc, yc, w, h = out[0]
        self.assertEqual(cls, 0)
        for got, want in zip((xc, yc, w, h), (0.95, 0.5, 0.1, 0.2)):
            self.assertAlmostEqual(got, want, places=6)

    def test_write_prelabel_does_not_overwrite(self):
        self._pre([(0, .5, .5, .1, .1, .9)])
        self.assertFalse(self.store.write_prelabel("test", STEM, [(2, .3, .3, .1, .1, .5)]))
        self.assertEqual(self.store.read_prelabel("test", STEM), [(0, .5, .5, .1, .1, .9)])

    def test_bad_stem_or_set_raises(self):
        with self.assertRaises(ValueError):
            self.store.save("test", "../x", [], "save")
        with self.assertRaises(ValueError):
            self.store.save("val", STEM, [], "save")

    def test_status_json_fields(self):
        self._pre([])
        self.store.save("test", STEM, [], "save")
        data = json.loads((self.store.set_dir("test") / "status" / f"{STEM}.json").read_text(encoding="utf-8"))
        self.assertEqual(set(data), {"status", "reviewed_at", "n_boxes"})

    def _corrupt_status(self):
        self._pre([])
        self.store.save("test", STEM, [], "save")
        bad = self.store.set_dir("test") / "status" / "cam0001_img00002.json"
        bad.write_text("<<<<<<< HEAD\n{\"status\": \"approved\"}\n=======\n", encoding="utf-8")
        return bad

    def test_bad_status_file_skipped_by_default(self):
        self._corrupt_status()
        self.assertEqual(self.store.all_status("test"), {STEM: "approved"})

    def test_bad_status_file_raises_when_strict(self):
        bad = self._corrupt_status()
        with self.assertRaises(ValueError) as ctx:
            self.store.all_status("test", strict=True)
        self.assertIn(bad.name, str(ctx.exception))

    def test_failed_write_leaves_no_temp_file(self):
        from unittest import mock
        self._pre([])
        with mock.patch("review_store.os.replace", side_effect=PermissionError("file in use")):
            with self.assertRaises(PermissionError):
                self.store.save("test", STEM, [], "save")
        leftovers = list(self.store.set_dir("test").rglob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_all_status_lists_saved_only(self):
        self._pre([])
        self.store.save("test", STEM, [], "save")
        self.assertEqual(self.store.all_status("test"), {STEM: "approved"})


if __name__ == "__main__":
    unittest.main()
