import unittest

import numpy as np
import pandas as pd

from DataAnalysis.rain.prepare import camera_hour, camera_table

M = 1 / 110_574  # degrees of latitude per metre


def counts_rows(cam, hour_ts, totals):
    """One frame per value in totals, 5 minutes apart inside the hour; all vehicles are cars."""
    t0 = pd.Timestamp(hour_ts)
    return [{"image_id": i, "camera_id": cam, "timestamp": (t0 + pd.Timedelta(minutes=5 * i)).isoformat(),
             "car": v, "motorcycle": 0, "bus": 0, "truck": 0, "total": v} for i, v in enumerate(totals)]


def cams(*ids):
    return pd.DataFrame({"camera_id": list(ids), "gauge": ["G1"] * len(ids)})


class CameraTableTest(unittest.TestCase):
    def test_camera_table_flags(self):
        c = pd.DataFrame({"camera_id": [1], "latitude": [13.7], "longitude": [100.5]})
        gauges = pd.DataFrame({"gauge": ["FAR", "NEAR"], "latitude": [13.8, 13.701], "longitude": [100.5, 100.5]})
        rail = pd.DataFrame({"latitude": [13.7 + 100 * M], "longitude": [100.5]})
        shelters = pd.DataFrame({"latitude": [13.7 - 120 * M], "longitude": [100.5]})
        t = camera_table(c, gauges, rail, shelters, pd.DataFrame(columns=["latitude", "longitude"])).iloc[0]
        self.assertEqual(t["gauge"], "NEAR")
        self.assertTrue(t["near_rail"])
        self.assertTrue(t["near_shelter"])
        self.assertFalse(t["in_flood_area"])
        self.assertAlmostEqual(t["rail_dist_m"], 100, delta=2)


class CameraHourTest(unittest.TestCase):
    def setUp(self):
        rows = (counts_rows(1, "2026-08-25 08:00", [1, 2, 3, 4, 5, 6])    # wet
                + counts_rows(1, "2026-08-26 08:00", [10])                # dry
                + counts_rows(1, "2026-08-27 08:00", [20])                # dry
                + counts_rows(1, "2026-08-25 09:00", [7])                 # wet, no dry 9:00 -> no baseline
                + counts_rows(1, "2026-08-28 08:00", [99]))               # no rain row -> dropped
        self.counts = pd.DataFrame(rows)
        self.rain = pd.DataFrame({"gauge": "G1", "rain_mm": [5.0, 0.9, 0.0, 1.0],
                                  "hour_ts": pd.to_datetime(["2026-08-25 08:00", "2026-08-26 08:00",
                                                             "2026-08-27 08:00", "2026-08-25 09:00"])})
        self.ch = camera_hour(self.counts, cams(1), self.rain).set_index("hour_ts")

    def test_hourly_mean_and_wet(self):
        r = self.ch.loc[pd.Timestamp("2026-08-25 08:00")]
        self.assertEqual(r["total"], 3.5)
        self.assertEqual(r["n_frames"], 6)
        self.assertTrue(r["wet"])
        self.assertTrue(r["heavy"])
        self.assertFalse(self.ch.loc[pd.Timestamp("2026-08-26 08:00"), "wet"])   # 0.9 mm

    def test_baseline_is_same_camera_same_hod_dry(self):
        self.assertEqual(self.ch.loc[pd.Timestamp("2026-08-25 08:00"), "total_base"], 15)

    def test_baseline_matches_weekday_or_weekend(self):
        # Sat 29 Aug wet at 08:00: baseline must come only from dry weekend 08:00 hours (Sun 30: 4), not weekdays (10, 20)
        rows = (counts_rows(1, "2026-08-29 08:00", [2]) + counts_rows(1, "2026-08-30 08:00", [4])
                + counts_rows(1, "2026-08-26 08:00", [10]) + counts_rows(1, "2026-08-27 08:00", [20]))
        rain = pd.DataFrame({"gauge": "G1", "rain_mm": [3.0, 0.0, 0.0, 0.0],
                             "hour_ts": pd.to_datetime(["2026-08-29 08:00", "2026-08-30 08:00",
                                                        "2026-08-26 08:00", "2026-08-27 08:00"])})
        ch = camera_hour(pd.DataFrame(rows), cams(1), rain).set_index("hour_ts")
        self.assertEqual(ch.loc[pd.Timestamp("2026-08-29 08:00"), "total_base"], 4)
        self.assertEqual(ch.loc[pd.Timestamp("2026-08-26 08:00"), "total_base"], 15)

    def test_missing_rain_hour_dropped(self):
        self.assertNotIn(pd.Timestamp("2026-08-28 08:00"), self.ch.index)

    def test_no_dry_baseline_dropped(self):
        self.assertNotIn(pd.Timestamp("2026-08-25 09:00"), self.ch.index)
        self.assertFalse(np.isinf(self.ch.select_dtypes("number")).any().any())
        self.assertFalse(self.ch[["total_base", "car_base", "motorcycle_base"]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
