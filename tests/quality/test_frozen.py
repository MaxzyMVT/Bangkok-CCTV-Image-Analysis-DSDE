import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "quality"))

from find_frozen_frames import flag_frozen  # noqa: E402


def _sig(v):
    return np.full((24, 32), v, dtype=np.float32)


class FlagFrozenTest(unittest.TestCase):
    def test_repeats_after_first_frame_are_frozen(self):
        # camera 1: live, then the same picture 3 times, then live again
        df = pd.DataFrame({"image_id": [1, 2, 3, 4, 5], "camera_id": [1] * 5,
                           "timestamp": [f"2026-08-25T0{h}:00:00" for h in range(5)]})
        sigs = {1: _sig(10), 2: _sig(50), 3: _sig(50), 4: _sig(50), 5: _sig(90)}
        out = flag_frozen(df, sigs).set_index("image_id")
        self.assertEqual(out["frozen"].tolist(), [False, False, True, True, False])
        self.assertEqual(out.loc[4, "frozen_since"], "2026-08-25T01:00:00")   # first copy of that picture
        self.assertEqual(out.loc[4, "run_length"], 3)
        self.assertTrue(pd.isna(out.loc[2, "frozen_since"]) or out.loc[2, "frozen_since"] is None)

    def test_cameras_and_time_order_are_separate(self):
        # rows out of order; same picture on two different cameras is not "frozen"
        df = pd.DataFrame({"image_id": [3, 1, 2], "camera_id": [2, 1, 1],
                           "timestamp": ["2026-08-25T00:00:00", "2026-08-25T01:00:00", "2026-08-25T00:00:00"]})
        sigs = {1: _sig(50), 2: _sig(50.2), 3: _sig(50)}
        out = flag_frozen(df, sigs, threshold=0.5).set_index("image_id")
        self.assertTrue(out.loc[1, "frozen"])       # 1 repeats 2 (same camera, later)
        self.assertFalse(out.loc[2, "frozen"])
        self.assertFalse(out.loc[3, "frozen"])      # other camera

    def test_small_change_is_live(self):
        # live frames differ at least by the burned-in clock; frozen ones are pixel-identical
        df = pd.DataFrame({"image_id": [1, 2], "camera_id": [1, 1],
                           "timestamp": ["2026-08-25T00:00:00", "2026-08-25T00:05:00"]})
        out = flag_frozen(df, {1: _sig(50), 2: _sig(52)}, threshold=0.5)
        self.assertFalse(out["frozen"].any())


if __name__ == "__main__":
    unittest.main()
