import unittest

import numpy as np
import pandas as pd

from DataAnalysis.rain.effects import city_effect, natural_experiment

M = 1 / 110_574
CLS = ["total", "car", "motorcycle"]


def row(cam, ts, ratio, rain, base=10.0):
    ts = pd.Timestamp(ts)
    r = {"camera_id": cam, "hour_ts": ts, "hod": ts.hour, "rain_mm": rain, "wet": rain >= 1.0}
    for c in CLS:
        r[c], r[c + "_base"] = base * ratio, base
    return r


def cams_at(lat_offsets_m):
    return pd.DataFrame({"camera_id": range(1, len(lat_offsets_m) + 1),
                         "latitude": [13.7 + m * M for m in lat_offsets_m], "longitude": [100.5] * len(lat_offsets_m)})


class CityEffectTest(unittest.TestCase):
    def setUp(self):
        rows = []
        for cam in range(1, 21):
            rows += [row(cam, "2026-08-25 08:00", 0.8, 3.0), row(cam, "2026-08-26 08:00", 1.0, 0.0)]
        self.t = city_effect(pd.DataFrame(rows)).set_index(["cls", "window"])

    def test_all_classes_drop_twenty_percent(self):
        for c in CLS:
            r = self.t.loc[(c, "all")]
            self.assertAlmostEqual(r["effect_pct"], -20, delta=0.5)
            self.assertLessEqual(r["ci_low"], -20 + 1e-9)
            self.assertGreaterEqual(r["ci_high"], -20 - 1e-9)
            self.assertEqual(r["n_cameras"], 20)

    def test_rush_split(self):
        self.assertAlmostEqual(self.t.loc[("total", "rush"), "effect_pct"], -20, delta=0.5)
        other = self.t.loc[("total", "other")]
        self.assertEqual(other["n_wet_hours"], 0)
        self.assertTrue(np.isnan(other["effect_pct"]))


class NaturalExperimentTest(unittest.TestCase):
    def test_wet_vs_dry_neighbours_same_hour(self):
        cams = cams_at([0, 1000, 2000, 4000])            # camera 4 is 4 km away
        ch = pd.DataFrame([row(1, "2026-08-25 17:00", 0.7, 4.0), row(2, "2026-08-25 17:00", 1.0, 0.0),
                           row(3, "2026-08-25 17:00", 1.0, 0.0), row(4, "2026-08-25 17:00", 3.0, 0.0)])
        t = natural_experiment(ch, cams).set_index("cls")
        self.assertAlmostEqual(t.loc["total", "effect_pct_points"], -30, delta=0.5)
        self.assertEqual(t.loc["total", "n_pairs"], 1)
        self.assertEqual(t.loc["total", "n_hours"], 1)

    def test_hour_without_dry_neighbour_skipped(self):
        cams = cams_at([0, 1000, 0, 1000])
        ch = pd.DataFrame([row(1, "2026-08-25 17:00", 0.7, 4.0), row(2, "2026-08-25 17:00", 1.0, 0.0),
                           row(3, "2026-08-26 17:00", 0.5, 4.0), row(4, "2026-08-26 17:00", 1.0, 0.5)])  # 0.5 mm: not dry
        t = natural_experiment(ch, cams).set_index("cls")
        self.assertEqual(t.loc["total", "n_pairs"], 1)
        self.assertAlmostEqual(t.loc["total", "effect_pct_points"], -30, delta=0.5)


if __name__ == "__main__":
    unittest.main()
