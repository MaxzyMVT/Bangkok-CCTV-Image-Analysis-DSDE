import unittest

import numpy as np
import pandas as pd

from DataAnalysis.rain.shelter import frame_stationary, is_stationary, median_background, pick_pairs, stationary_share


def frame(with_square: bool) -> np.ndarray:
    img = np.full((40, 40), 60, np.uint8)
    if with_square:
        img[10:20, 10:20] = 220
    return img


BOX = (0.375, 0.375, 0.25, 0.25)   # x, y, w, h (normalised) over the 10:20 square


class StationaryTest(unittest.TestCase):
    def test_square_in_all_frames_is_background(self):
        frames = [frame(True) for _ in range(5)]
        self.assertTrue(is_stationary(frames[0], median_background(frames), BOX))

    def test_square_in_one_frame_is_moving(self):
        frames = [frame(True)] + [frame(False) for _ in range(4)]
        self.assertFalse(is_stationary(frames[0], median_background(frames), BOX))


class FrameStationaryTest(unittest.TestCase):
    def test_odd_sized_frame_skipped_not_crashing(self):
        counts = pd.DataFrame({"image_id": [1, 2, 3, 4], "camera_id": 7,
                               "timestamp": [f"2026-08-25T08:{m:02d}:00" for m in (0, 10, 20, 30)]})
        boxes = pd.DataFrame({"image_id": [1], "cls": [1], "x": [.375], "y": [.375], "w": [.25], "h": [.25]})
        imgs = {1: frame(True), 2: frame(True), 3: frame(True), 4: np.zeros((30, 50), np.uint8)}
        f = frame_stationary(counts, boxes, [7], imgs.get)
        self.assertEqual(sorted(f["image_id"]), [1, 2, 3])
        self.assertEqual(f.set_index("image_id").loc[1, "stationary_moto"], 1)


class ShareAndPairsTest(unittest.TestCase):
    def setUp(self):
        self.frames = pd.DataFrame({
            "image_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "camera_id": [10, 10, 10, 10, 20, 20, 20, 20],
            "timestamp": ["2026-08-25T08:00:00", "2026-08-25T08:10:00", "2026-08-26T08:00:00", "2026-08-26T08:10:00"] * 2,
            "moto_boxes": [4, 4, 4, 4, 4, 4, 4, 4],
            "stationary_moto": [0, 1, 3, 4, 1, 1, 2, 1]})
        self.ch = pd.DataFrame({"camera_id": [10, 10, 20, 20],
                                "hour_ts": pd.to_datetime(["2026-08-25 08:00", "2026-08-26 08:00"] * 2),
                                "wet": [False, True, False, True]})

    def test_share_per_camera_and_weather(self):
        s = stationary_share(self.frames, self.ch).set_index(["camera_id", "wet"])
        self.assertAlmostEqual(s.loc[(10, False), "share"], 1 / 8)
        self.assertAlmostEqual(s.loc[(10, True), "share"], 7 / 8)

    def test_pick_pairs_biggest_rise_first(self):
        s = stationary_share(self.frames, self.ch)
        p = pick_pairs(self.frames, self.ch, s, k=2)
        self.assertEqual(p["camera_id"].tolist(), [10, 20])
        self.assertEqual((p.iloc[0]["dry_image_id"], p.iloc[0]["wet_image_id"]), (2, 4))


if __name__ == "__main__":
    unittest.main()
