import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Models" / "training"))

from train_pipeline import build_image_list, stretch_copy  # noqa: E402


class BuildImageListTest(unittest.TestCase):
    def test_bus_frames_listed_n_times_others_once(self):
        entries = [("./images/train/a.jpg", "0 0.5 0.5 0.1 0.1\n"),
                   ("./images/train/b.jpg", "2 0.5 0.5 0.3 0.3\n0 0.1 0.1 0.1 0.1\n"),
                   ("./images/train/c.jpg", "3 0.5 0.5 0.3 0.3\n"),
                   ("./images/train/d.jpg", "")]
        out = build_image_list(entries, oversample_bus=3)
        self.assertEqual(out.count("./images/train/a.jpg"), 1)
        self.assertEqual(out.count("./images/train/b.jpg"), 3)
        self.assertEqual(out.count("./images/train/c.jpg"), 1)  # trucks are not up-weighted
        self.assertEqual(out.count("./images/train/d.jpg"), 1)
        self.assertEqual(len(out), 6)

    def test_oversample_one_lists_each_frame_once(self):
        entries = [("x.jpg", "2 0.5 0.5 0.3 0.3\n"), ("y.jpg", "0 0.5 0.5 0.1 0.1\n")]
        self.assertEqual(build_image_list(entries, oversample_bus=1), ["x.jpg", "y.jpg"])


class StretchCopyTest(unittest.TestCase):
    def test_stretched_copy_changes_width_only_and_keeps_labels(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            Image.new("RGB", (352, 288), (40, 80, 120)).save(d / "a.jpg")
            label = "0 0.500000 0.500000 0.100000 0.100000\n2 0.250000 0.400000 0.200000 0.300000\n"
            (d / "a.txt").write_text(label, encoding="utf-8")
            for factor in (0.8, 1.2):
                img_dst, lbl_dst = d / f"a_{factor}.jpg", d / f"a_{factor}.txt"
                stretch_copy(d / "a.jpg", d / "a.txt", img_dst, lbl_dst, factor)
                with Image.open(img_dst) as im:
                    self.assertEqual(im.size, (round(352 * factor), 288))
                self.assertEqual(lbl_dst.read_text(encoding="utf-8").splitlines(), label.splitlines())


if __name__ == "__main__":
    unittest.main()
