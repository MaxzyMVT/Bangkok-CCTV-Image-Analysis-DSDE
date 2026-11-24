import unittest

import pandas as pd

from DataAnalysis.rain.stratified import rain_timing, stratified_effects


class StratifiedEffectsTest(unittest.TestCase):
    def test_weekday_weekend_use_matching_baselines_and_clock_hours(self):
        rows = pd.DataFrame([
            {"camera_id": 1, "hour_ts": "2026-08-25 08:00:00", "weekend": False,
             "wet": True, "rain_mm": 2, "total": 5, "total_base": 10,
             "car": 3, "car_base": 6, "motorcycle": 2, "motorcycle_base": 4},
            {"camera_id": 2, "hour_ts": "2026-08-25 08:00:00", "weekend": False,
             "wet": True, "rain_mm": 2, "total": 5, "total_base": 10,
             "car": 3, "car_base": 6, "motorcycle": 2, "motorcycle_base": 4},
            {"camera_id": 1, "hour_ts": "2026-08-29 08:00:00", "weekend": True,
             "wet": True, "rain_mm": 6, "total": 10, "total_base": 10,
             "car": 6, "car_base": 6, "motorcycle": 4, "motorcycle_base": 4},
        ])
        result = stratified_effects(rows).set_index(["segment", "cls"])
        self.assertEqual(result.loc[("weekday", "motorcycle"), "effect_pct"], -50)
        self.assertEqual(result.loc[("weekend", "motorcycle"), "effect_pct"], 0)
        self.assertEqual(result.loc[("weekday", "motorcycle"), "n_clock_hours"], 1)
        timing = rain_timing(rows).set_index(["day_type", "hour"])
        self.assertEqual(timing.loc[("weekday", 8), "n_camera_hours"], 2)
        self.assertEqual(timing.loc[("weekday", 8), "n_clock_hours"], 1)


if __name__ == "__main__":
    unittest.main()
