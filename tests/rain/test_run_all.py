import tempfile
import unittest
from pathlib import Path

import pandas as pd

from DataAnalysis.rain.run_all import write_tables


class WriteTablesTest(unittest.TestCase):
    def test_coordinates_keep_full_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tables({"camera": pd.DataFrame({"camera_id": [11], "latitude": [13.76363], "longitude": [100.54106],
                                                  "effect_pct": [-5.4321]})}, Path(tmp))
            back = pd.read_csv(Path(tmp) / "camera.csv")
        self.assertAlmostEqual(back.loc[0, "longitude"], 100.54106, places=5)
        self.assertAlmostEqual(back.loc[0, "latitude"], 13.76363, places=5)


if __name__ == "__main__":
    unittest.main()
