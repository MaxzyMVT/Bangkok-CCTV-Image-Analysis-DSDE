import unittest
from pathlib import Path

import pandas as pd

from DataAnalysis.rain.recount import frame_rows, recount


class FrameRowsTest(unittest.TestCase):
    def test_frame_rows_counts_classes(self):
        boxes = [(0, .1, .1, .1, .1, .9), (0, .3, .3, .1, .1, .8), (1, .5, .5, .1, .1, .7), (3, .7, .7, .1, .1, .6)]
        self.assertEqual(frame_rows(boxes), {"car": 2, "motorcycle": 1, "bus": 0, "truck": 1, "total": 4})


class RecountTest(unittest.TestCase):
    def test_recount_uses_predict_and_keeps_boxes(self):
        frames = pd.DataFrame({"image_id": [11, 12], "camera_id": [5, 5],
                               "timestamp": ["2026-08-25T08:00:00", "2026-08-25T08:10:00"],
                               "path": [Path("a.jpg"), Path("b.jpg")]})
        fake = {Path("a.jpg"): [(0, .2, .2, .1, .1, .9), (1, .6, .6, .1, .1, .5)], Path("b.jpg"): []}
        counts, boxes = recount(frames, lambda p: fake[p])
        self.assertEqual(len(counts), 2)
        self.assertEqual(counts.set_index("image_id").loc[11, "total"], 2)
        self.assertEqual(counts.set_index("image_id").loc[12, ["car", "motorcycle", "bus", "truck", "total"]].sum(), 0)
        self.assertEqual(list(boxes["image_id"]), [11, 11])
        self.assertEqual(list(boxes.columns), ["image_id", "cls", "x", "y", "w", "h", "conf"])


if __name__ == "__main__":
    unittest.main()
