import unittest

import pandas as pd

from DataAnalysis.rain.common import bootstrap_ci, haversine_m, nearest, within


class HaversineTest(unittest.TestCase):
    def test_one_degree_latitude(self):
        self.assertAlmostEqual(float(haversine_m(13, 100, 14, 100)), 110_574, delta=1_106)

    def test_zero_distance(self):
        self.assertEqual(float(haversine_m(13.7, 100.5, 13.7, 100.5)), 0.0)


class NearestTest(unittest.TestCase):
    def test_picks_closest(self):
        pts = pd.DataFrame({"code": ["far", "near", "mid"], "latitude": [13.8, 13.7001, 13.71], "longitude": [100.5] * 3})
        key, dist = nearest(13.7, 100.5, pts, "code")
        self.assertEqual(key, "near")
        self.assertAlmostEqual(dist, 11.06, delta=0.5)


class WithinTest(unittest.TestCase):
    def test_radius(self):
        deg_per_m = 1 / 110_574
        pts = pd.DataFrame({"latitude": [13.7 + 140 * deg_per_m], "longitude": [100.5]})
        self.assertTrue(within(13.7, 100.5, pts, 150))
        pts = pd.DataFrame({"latitude": [13.7 + 160 * deg_per_m], "longitude": [100.5]})
        self.assertFalse(within(13.7, 100.5, pts, 150))

    def test_empty_points(self):
        self.assertFalse(within(13.7, 100.5, pd.DataFrame({"latitude": [], "longitude": []}), 150))


class BootstrapTest(unittest.TestCase):
    def test_ci_brackets_mean(self):
        groups = [[v] for v in range(1, 11)]
        stat = lambda gs: sum(x for g in gs for x in g) / sum(len(g) for g in gs)
        low, high = bootstrap_ci(groups, stat)
        self.assertLess(low, 5.5)
        self.assertGreater(high, 5.5)
        self.assertEqual((low, high), bootstrap_ci(groups, stat))


if __name__ == "__main__":
    unittest.main()
