import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Models"))

from pipeline import DEFAULT_FROZEN_CSV, drop_frozen, load_frozen_keys  # noqa: E402

CSV = ("\ufeffstem,image_id,camera_id,timestamp,mean_diff,max_diff,frozen,frozen_since,run_length,unreadable\n"
       "cam0150_img00002,2,150,2026-08-27T07:46:03,0.0,0.0,True,2026-08-27T07:36:03,2,False\n")


class FrozenFilterTest(unittest.TestCase):
    def test_frozen_frames_are_skipped_by_camera_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "frozen.csv"
            csv_path.write_text(CSV, encoding="utf-8")
            keys = load_frozen_keys(csv_path)
        paths = [Path("150_แยกA_รถมาจากB_20260827_073603.jpg"),   # first frame of the run: real, kept
                 Path("150_แยกA_รถมาจากB_20260827_074603.jpg"),   # frozen copy: skipped
                 Path("15_แยกC_20260827_074603.jpg"),             # other camera, same time: kept
                 Path("not_a_cctv_name.jpg")]                     # unparsable: kept
        kept, skipped = drop_frozen(paths, keys)
        self.assertEqual(skipped, 1)
        self.assertEqual([p.name for p in kept], [paths[0].name, paths[2].name, paths[3].name])

    def test_missing_file_means_nothing_skipped(self):
        self.assertEqual(load_frozen_keys(Path("does/not/exist.csv")), set())

    def test_default_points_at_committed_list(self):
        self.assertTrue(DEFAULT_FROZEN_CSV.exists())
        self.assertGreater(len(load_frozen_keys(DEFAULT_FROZEN_CSV)), 500)


if __name__ == "__main__":
    unittest.main()
