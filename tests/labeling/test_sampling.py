import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "DataEngineering" / "labeling" / "sample_test_candidates.py"
# Last commit before the member split was removed: the frozen test-set stems
FROZEN_REV = "ba6fc25"


class SamplingTest(unittest.TestCase):
    def test_candidates_have_no_member_and_same_stems(self):
        frozen = subprocess.run(["git", "show", f"{FROZEN_REV}:DataEngineering/test_set/candidates.csv"],
                                cwd=ROOT, capture_output=True, check=True).stdout
        frozen_stems = set(pd.read_csv(io.BytesIO(frozen), encoding="utf-8-sig")["stem"])
        self.assertEqual(len(frozen_stems), 1000)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidates.csv"
            subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], cwd=ROOT, check=True,
                           capture_output=True)
            df = pd.read_csv(out, encoding="utf-8-sig")
        self.assertNotIn("member", df.columns)
        self.assertEqual(set(df["stem"]), frozen_stems)


class CoreSubsetTest(unittest.TestCase):
    """300-frame core subset: what one reviewer can finish; 3 frames per camera, 3 different periods."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPT.parent))
        from common import CANDIDATES_CSV
        from sample_test_candidates import pick_core
        cls.pick_core = staticmethod(pick_core)
        cls.cand = pd.read_csv(CANDIDATES_CSV, encoding="utf-8-sig")

    def test_three_per_camera_in_different_periods(self):
        core = self.cand[self.pick_core(self.cand, per_camera=3, seed=42)]
        self.assertEqual(len(core), 300)
        self.assertTrue((core.groupby("camera_id").size() == 3).all())
        self.assertTrue((core.groupby("camera_id")["time_period"].nunique() == 3).all())
        # periods rotate over cameras, so each of the 5 periods gets ~60 frames
        per_period = core["time_period"].value_counts()
        self.assertEqual(len(per_period), 5)
        self.assertLessEqual(per_period.max() - per_period.min(), 6)

    def test_deterministic_and_committed(self):
        a = self.pick_core(self.cand, per_camera=3, seed=42)
        b = self.pick_core(self.cand, per_camera=3, seed=42)
        self.assertTrue((a == b).all())
        self.assertIn("core", self.cand.columns)
        self.assertEqual(list(self.cand["core"].astype(bool)), list(a))


if __name__ == "__main__":
    unittest.main()
