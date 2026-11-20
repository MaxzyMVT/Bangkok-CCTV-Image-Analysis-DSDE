import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "labeling"))

from assign_reviewers import assign  # noqa: E402

NAMES = ["Pan", "Phat", "Cream"]


def _frames(sizes):
    """sizes: frames per camera -> DataFrame(stem, camera_id)."""
    rows = [{"stem": f"cam{c:04d}_img{c * 100 + i:05d}", "camera_id": c}
            for c, n in enumerate(sizes, start=1) for i in range(n)]
    return pd.DataFrame(rows)


class AssignTest(unittest.TestCase):
    def test_every_frame_assigned_once_and_whole_cameras(self):
        df = _frames([10, 9, 7, 7, 6, 5, 5, 3, 2, 1])
        out = assign(df, NAMES)
        self.assertEqual(len(out), len(df))
        self.assertEqual(set(out["stem"]), set(df["stem"]))
        self.assertTrue((out.groupby("camera_id")["reviewer"].nunique() == 1).all())

    def test_shares_are_balanced(self):
        df = _frames([10, 9, 8, 8, 7, 6, 6, 5, 5, 4, 3, 3, 2, 1, 1] * 6)
        counts = assign(df, NAMES)["reviewer"].value_counts()
        self.assertEqual(set(counts.index), set(NAMES))
        self.assertLessEqual(counts.max() - counts.min(), 10)   # never more than one camera apart

    def test_deterministic(self):
        df = _frames([4, 8, 3, 9, 1, 6, 2])
        pd.testing.assert_frame_equal(assign(df, NAMES), assign(df.sample(frac=1, random_state=1), NAMES))


if __name__ == "__main__":
    unittest.main()
