import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "WebScraping"))

from metar import weather_flags  # noqa: E402


class WeatherFlagsTest(unittest.TestCase):
    def test_flags_follow_metar_weather_codes(self):
        flags = weather_flags(pd.Series(["-RA", "+TSRA", "VCSH", "", None]))
        self.assertEqual(flags["is_rain"].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(flags["is_heavy_rain"].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(flags["is_thunderstorm"].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(flags["is_shower"].tolist(), [0, 0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
