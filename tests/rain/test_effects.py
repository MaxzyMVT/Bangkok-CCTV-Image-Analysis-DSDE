import unittest

import numpy as np
import pandas as pd

from DataAnalysis.rain.effects import camera_effect, city_effect, denser_share, mealtime, natural_experiment, rebound

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


class CameraEffectTest(unittest.TestCase):
    def test_labels(self):
        rows = []
        for i in range(5):
            rows += [row(1, f"2026-08-2{i + 1} 08:00", 1.5, 3.0), row(2, f"2026-08-2{i + 1} 08:00", 0.5, 3.0)]
        rows += [row(3, "2026-08-25 08:00", 0.5, 3.0), row(3, "2026-08-26 08:00", 0.5, 3.0)]
        t = camera_effect(pd.DataFrame(rows)).set_index("camera_id")
        self.assertEqual(t.loc[1, "label"], "denser")
        self.assertAlmostEqual(t.loc[1, "effect_pct"], 50, delta=0.5)
        self.assertEqual(t.loc[2, "label"], "clears")
        self.assertEqual(t.loc[3, "label"], "too few wet hours")
        self.assertEqual(t.loc[3, "n_wet_hours"], 2)


class DenserShareTest(unittest.TestCase):
    def test_share_difference(self):
        eff = pd.DataFrame({"camera_id": range(1, 12), "label": ["denser"] * 4 + ["clears"] * 6 + ["too few wet hours"]})
        cams = pd.DataFrame({"camera_id": range(1, 12), "near_rail": [True] * 4 + [False] * 6 + [True],
                             "near_shelter": [False] * 11, "in_flood_area": [False] * 11})
        t = denser_share(eff, cams).set_index("feature")
        self.assertEqual(t.loc["near_rail", "denser_share"], 1.0)
        self.assertEqual(t.loc["near_rail", "other_share"], 0.0)
        self.assertEqual(t.loc["near_rail", "diff_points"], 100)
        self.assertEqual((t.loc["near_rail", "n_denser"], t.loc["near_rail", "n_other"]), (4, 6))


class MealtimeTest(unittest.TestCase):
    def test_effect_by_hour(self):
        rows = []
        for cam in range(1, 6):
            rows += [row(cam, "2026-08-25 12:00", 0.9, 3.0), row(cam, "2026-08-25 15:00", 0.5, 3.0)]
        t = mealtime(pd.DataFrame(rows)).set_index("hod")
        self.assertAlmostEqual(t.loc[12, "effect_pct"], -10, delta=0.5)
        self.assertAlmostEqual(t.loc[15, "effect_pct"], -50, delta=0.5)
        self.assertTrue(t.loc[12, "meal"])
        self.assertFalse(t.loc[15, "meal"])


def frames(cam, start, end_excl, value):
    ts = pd.date_range(start, end_excl, freq="10min", inclusive="left")
    return pd.DataFrame({"camera_id": cam, "timestamp": [t.isoformat() for t in ts],
                         "motorcycle": value, "car": value, "total": value})


class ReboundTest(unittest.TestCase):
    def setUp(self):
        self.ch = pd.DataFrame([row(1, "2026-08-25 08:00", 0.5, 3.0), row(1, "2026-08-25 09:00", 1.3, 0.0),
                                row(1, "2026-08-25 10:00", 1.3, 0.0)])

    def test_after_rain_level(self):
        counts = pd.concat([frames(1, "2026-08-25 08:00", "2026-08-25 09:00", 5.0),
                            frames(1, "2026-08-25 09:00", "2026-08-25 11:00", 13.0)])
        t = rebound(counts, self.ch).set_index(["cls", "minute"])
        self.assertAlmostEqual(t.loc[("motorcycle", 30), "ratio"], 1.3, delta=1e-9)
        self.assertAlmostEqual(t.loc[("motorcycle", -30), "ratio"], 0.5, delta=1e-9)
        self.assertEqual(t.loc[("motorcycle", 30), "n_events"], 1)
        self.assertEqual(sorted(t.loc["car"].index), list(range(-60, 120, 10)))

    def test_rebound_event_at_data_end(self):
        counts = frames(1, "2026-08-25 08:00", "2026-08-25 09:00", 5.0)          # nothing after t0 = 09:00
        t = rebound(counts, self.ch).set_index(["cls", "minute"])
        self.assertTrue(np.isnan(t.loc[("motorcycle", 0), "ratio"]))
        self.assertEqual(t.loc[("motorcycle", 0), "n_events"], 0)
        self.assertEqual(t.loc[("motorcycle", -10), "n_events"], 1)


if __name__ == "__main__":
    unittest.main()
