import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "labeling"))

from review_store import ReviewStore, clean_boxes, counts  # noqa: E402

STEM = "cam0001_img00001"


class ReviewStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ReviewStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _pre(self, boxes):
        self.store.write_prelabel("test", STEM, boxes)

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

    def test_all_status_lists_saved_only(self):
        self._pre([])
        self.store.save("test", STEM, [], "save")
        self.assertEqual(self.store.all_status("test"), {STEM: "approved"})


if __name__ == "__main__":
    unittest.main()
