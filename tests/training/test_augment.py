import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Models" / "training"))

from train_pipeline import AUGMENT  # noqa: E402


class AugmentTest(unittest.TestCase):
    def test_light_geometric_distortion(self):
        # small shear/perspective: camera angles differ a little between sites
        self.assertEqual(AUGMENT["shear"], 2.0)
        self.assertEqual(AUGMENT["perspective"], 0.0005)
        self.assertEqual(AUGMENT["degrees"], 0.0)  # no rotation: horizons stay level

    def test_values_within_ultralytics_ranges(self):
        self.assertLessEqual(AUGMENT["perspective"], 0.001)
        self.assertLessEqual(abs(AUGMENT["shear"]), 10)


if __name__ == "__main__":
    unittest.main()
