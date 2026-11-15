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


if __name__ == "__main__":
    unittest.main()
