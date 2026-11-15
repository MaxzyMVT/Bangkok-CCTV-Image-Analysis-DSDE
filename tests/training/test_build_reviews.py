import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Models" / "training"))

from train_pipeline import count_mae, select_frames  # noqa: E402


def _frames(ids):
    return pd.DataFrame({"image_id": ids, "camera_id": [1] * len(ids), "date": ["2026-08-25"] * len(ids),
                         "stem": [f"cam0001_img{i:05d}" for i in ids]})


class SelectFramesTest(unittest.TestCase):
    def setUp(self):
        self.pool = _frames([1, 2, 3, 4, 5])
        self.picked = _frames([1, 2, 3])

    def test_rejected_frames_dropped(self):
        out = select_frames(self.picked, self.pool, {"cam0001_img00002": "rejected"})
        self.assertNotIn("cam0001_img00002", set(out["stem"]))
        self.assertEqual(len(out), 2)

    def test_reviewed_frames_forced_in(self):
        out = select_frames(self.picked, self.pool, {"cam0001_img00005": "corrected", "cam0001_img00001": "approved"})
        self.assertIn("cam0001_img00005", set(out["stem"]))
        src = dict(zip(out["stem"], out["label_source"]))
        self.assertEqual(src["cam0001_img00005"], "human")
        self.assertEqual(src["cam0001_img00001"], "human")
        self.assertEqual(src["cam0001_img00003"], "pseudo")
        self.assertEqual(len(out), 4)


class CountMaeTest(unittest.TestCase):
    def test_count_mae_per_class(self):
        model = [{"car": 3, "motorcycle": 1, "bus": 0, "truck": 0, "total": 4},
                 {"car": 1, "motorcycle": 0, "bus": 0, "truck": 1, "total": 2}]
        truth = [{"car": 5, "motorcycle": 1, "bus": 0, "truck": 0, "total": 6},
                 {"car": 1, "motorcycle": 2, "bus": 0, "truck": 0, "total": 3}]
        m = count_mae(model, truth)
        self.assertAlmostEqual(m["car"], 1.0)
        self.assertAlmostEqual(m["motorcycle"], 1.0)
        self.assertAlmostEqual(m["truck"], 0.5)
        self.assertAlmostEqual(m["total"], 1.5)


if __name__ == "__main__":
    unittest.main()
