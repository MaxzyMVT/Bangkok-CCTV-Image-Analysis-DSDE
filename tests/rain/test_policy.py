import unittest

import pandas as pd

from DataAnalysis.rain.policy import policy_stations

M = 1 / 110_574


class PolicyStationsTest(unittest.TestCase):
    def test_station_near_denser_camera_ranked_by_lost_motorcycles(self):
        cams = pd.DataFrame({"camera_id": [1, 2], "latitude": [13.7, 13.8], "longitude": [100.5, 100.5]})
        eff = pd.DataFrame({"camera_id": [1, 2], "label": ["denser", "clears"]})
        ch = pd.DataFrame({"camera_id": [1, 1, 1, 2], "wet": [True, True, False, True],
                           "motorcycle": [6.0, 8.0, 10.0, 1.0], "motorcycle_base": [10.0, 12.0, 10.0, 9.0]})
        rail = pd.DataFrame({"name": ["Near", "Far", "ByClears"], "latitude": [13.7 + 300 * M, 13.7 + 900 * M, 13.8],
                             "longitude": [100.5] * 3})
        t = policy_stations(eff, ch, cams, rail)
        self.assertEqual(t["station"].tolist(), ["Near"])
        self.assertEqual(t.iloc[0]["moto_lost_per_wet_hour"], 4.0)
        self.assertEqual(t.iloc[0]["denser_cameras"], 1)
        self.assertIn("1", str(t.iloc[0]["camera_ids"]))


if __name__ == "__main__":
    unittest.main()
