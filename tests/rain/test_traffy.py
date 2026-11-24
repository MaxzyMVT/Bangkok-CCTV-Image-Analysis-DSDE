import unittest

import numpy as np
import pandas as pd

from DataAnalysis.rain.traffy import flood_area_table, flood_complaints, flood_points, flood_sensor_hours

M = 1 / 110_574


def traffy(lats, lons, flags=None, desc=None):
    d = pd.DataFrame({"latitude": lats, "longitude": lons, "timestamp": "2026-08-25 10:00:00",
                      "description": desc if desc is not None else ["x"] * len(lats)})
    if flags is not None:
        d["is_flood"] = flags
    return d


class FloodComplaintsTest(unittest.TestCase):
    def test_uses_is_flood_flag(self):
        self.assertEqual(len(flood_complaints(traffy([13.7, 13.71], [100.5, 100.5], flags=[1, 0]))), 1)

    def test_keyword_fallback_without_flag(self):
        t = traffy([13.7, 13.71], [100.5, 100.5], desc=["ถนนน้ำท่วมขัง", "ไฟดับ"])
        self.assertEqual(len(flood_complaints(t)), 1)

    def test_blank_description_and_missing_coords_ignored(self):
        t = traffy([13.7, np.nan, 13.72], [100.5, 100.5, None], desc=[None, "น้ำท่วม", "น้ำท่วม"])
        self.assertEqual(len(flood_complaints(t)), 0)          # row 1: no text; rows 2-3: no coordinates
        t = traffy([13.7, np.nan], [100.5, 100.5], flags=[1, 1])
        self.assertEqual(len(flood_complaints(t)), 1)


class FloodPointsTest(unittest.TestCase):
    def test_clusters_dense_group_drops_noise(self):
        lats = [13.7 + i * 8 * M for i in range(6)] + [13.9]
        pts = flood_points(traffy(lats, [100.5] * 7, flags=[1] * 7))
        self.assertEqual(len(pts), 6)
        self.assertEqual(pts["cluster"].nunique(), 1)
        table = flood_area_table(pts)
        self.assertEqual(table["n_complaints"].tolist(), [6])


class FloodSensorTest(unittest.TestCase):
    def test_flood_sensor_hours(self):
        f = pd.DataFrame({"station_code": ["FL.A", "FL.B"], "road": ["r1", "r2"], "station": ["s1", "s2"],
                          "datetime": pd.to_datetime(["2026-08-29 17:00", "2026-08-29 18:00"]), "flood_cm": [3.0, 8.0]})
        out = flood_sensor_hours(f)
        self.assertEqual(out["flood_cm"].tolist(), [8.0])
        self.assertEqual(list(out.columns), ["station_code", "road", "station", "hour_ts", "flood_cm"])


if __name__ == "__main__":
    unittest.main()
