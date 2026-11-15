import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "labeling"))

from merge_test_set import count_error_table, merge  # noqa: E402
from review_store import ReviewStore  # noqa: E402

STEMS = ["cam0001_img00001", "cam0001_img00002", "cam0001_img00003", "cam0001_img00004"]


def _counts(car=0, moto=0, bus=0, truck=0):
    return {"car": car, "motorcycle": moto, "bus": bus, "truck": truck, "total": car + moto + bus + truck}


class CountErrorTest(unittest.TestCase):
    def test_count_error_signed_and_mae(self):
        rows = [{"stem": "a", "time_period": "Midday", "pre": _counts(car=5), "final": _counts(car=7)},
                {"stem": "b", "time_period": "Midday", "pre": _counts(car=3), "final": _counts(car=3)}]
        t = count_error_table(rows)
        car_all = t[(t["class"] == "car") & (t["time_period"] == "all")].iloc[0]
        self.assertEqual(car_all["n_frames"], 2)
        self.assertAlmostEqual(car_all["mae"], 1.0)
        self.assertAlmostEqual(car_all["mean_signed_error"], -1.0)
        self.assertIn("Midday", set(t["time_period"]))
        self.assertIn("total", set(t["class"]))


class MergeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.store = ReviewStore(tmp / "reviews")
        self.data = tmp / "data"
        self.out = tmp / "out"
        rows = []
        for s in STEMS:
            rel = f"dataset/cctv_images/1/{s}.jpg"
            (self.data / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.data / rel).write_bytes(b"\xff\xd8fakejpeg")
            self.store.write_prelabel("test", s, [(0, .5, .5, .1, .1, .9)])
            rows.append({"stem": s, "image_id": 1, "camera_id": 1, "timestamp": "2026-08-25T08:00:00",
                         "time_period": "Morning Rush", "relative_path": rel})
        self.items = pd.DataFrame(rows)
        self.store.save("test", STEMS[0], [(0, .5, .5, .1, .1)], "save")                       # approved
        self.store.save("test", STEMS[1], [(0, .5, .5, .1, .1), (1, .2, .2, .05, .1)], "save")  # corrected
        self.store.save("test", STEMS[2], [], "reject")                                        # rejected
        # STEMS[3] stays pending

    def tearDown(self):
        self._tmp.cleanup()

    def test_merge_keeps_approved_and_corrected_only(self):
        summary = merge(self.store, self.items, self.data, self.out, allow_unreviewed=True)
        self.assertEqual(sorted(p.stem for p in (self.out / "labels").glob("*.txt")), STEMS[:2])
        self.assertEqual(sorted(p.stem for p in (self.out / "images").glob("*.jpg")), STEMS[:2])
        self.assertEqual(summary["kept"], 2)
        manifest = pd.read_csv(self.out / "test_manifest.csv", encoding="utf-8-sig")
        self.assertEqual(sorted(manifest["status"]), ["approved", "corrected"])
        self.assertTrue((self.out / "count_error.csv").exists())
        self.assertTrue((self.out / "dataset.yaml").exists())

    def test_merge_refuses_with_pending(self):
        with self.assertRaises(SystemExit):
            merge(self.store, self.items, self.data, self.out)


if __name__ == "__main__":
    unittest.main()
