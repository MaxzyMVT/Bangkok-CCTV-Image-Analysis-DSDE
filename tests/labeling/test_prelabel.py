import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "labeling"))

from prelabel import load_items, prelabel_one, to_prelabel_rows  # noqa: E402
from review_store import ReviewStore  # noqa: E402

STEM = "cam0001_img00001"


class PrelabelTest(unittest.TestCase):
    def test_to_prelabel_rows_maps_coco(self):
        rows = to_prelabel_rows([2, 3, 5, 7, 0], [[.5, .5, .1, .1]] * 5, [.9] * 5)
        self.assertEqual([r[0] for r in rows], [0, 1, 2, 3])
        self.assertTrue(all(r[5] == .9 for r in rows))

    def test_load_items_train_has_paths(self):
        items = load_items("train")
        self.assertEqual(len(items), 1200)
        self.assertFalse(items["relative_path"].isna().any())
        for col in ("stem", "camera_id", "timestamp", "time_period"):
            self.assertIn(col, items.columns)

    def test_load_items_test(self):
        self.assertEqual(len(load_items("test")), 1000)

    def test_rerun_skips_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewStore(Path(tmp))
            store.save("test", STEM, [(1, .5, .5, .1, .1)], "save")
            before = store.read_labels("test", STEM)
            self.assertEqual(prelabel_one(store, "test", STEM, [(0, .2, .2, .1, .1, .8)]), "skipped")
            self.assertEqual(store.read_labels("test", STEM), before)
            self.assertEqual(store.read_prelabel("test", STEM), [])

    def test_writes_when_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewStore(Path(tmp))
            self.assertEqual(prelabel_one(store, "test", STEM, [(0, .2, .2, .1, .1, .8)]), "written")
            self.assertEqual(prelabel_one(store, "test", STEM, [(0, .2, .2, .1, .1, .8)]), "skipped")


if __name__ == "__main__":
    unittest.main()
