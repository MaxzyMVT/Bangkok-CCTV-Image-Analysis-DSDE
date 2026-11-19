import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "WebScraping"))

from metar import parse_day, relative_humidity, weather_flags  # noqa: E402

OBHISTORY_HTML = """
<html><body><table>
<tr align="center"><th rowspan="3">Time</th><th>Wind (mph)</th></tr>
<tr style="background: #EEE;" class="ob" data-madis="0"></div><td>3:00 PM</td><td>WSW 28</td><td>0.93</td>
<td>SCT020<br />BKN040<br /></td><td>+TSRA</td><td>79</td><td>73</td><td>79</td><td></td><td></td><td>84</td>
<td>29.73661</td><td></td><td></td><td>0.00</td><td>M</td><td>M</td></tr>
<tr style="background: #FFF;" class="metar"><td colspan="17">VTBD 290800Z 25024KT 220V280 1500 +TSRA SCT020CB BKN040 26/23 Q1007 BECMG FM0815 TL0830 3000 TSRA</td></tr>
<tr style="background: #EEE;" class="ob" data-madis="0"></div><td>3:30 PM</td><td>Calm</td><td>6.21</td>
<td>FEW020<br /></td><td></td><td>75</td><td>73</td><td>75</td><td></td><td></td><td>94</td>
<td>29.70708</td><td></td><td></td><td>0.00</td><td>M</td><td>M</td></tr>
<tr style="background: #FFF;" class="metar"><td colspan="17">VTBD 290830Z 00000KT 9999 FEW020 24/23 Q1006 NOSIG</td></tr>
</table></body></html>
"""


class ParseDayTest(unittest.TestCase):
    def test_reads_rows_from_malformed_observation_table(self):
        rows = parse_day(OBHISTORY_HTML, "VTBD", "2026-08-29")
        self.assertEqual(len(rows), 2)
        storm, calm = rows
        self.assertEqual(storm["datetime"], datetime(2026, 8, 29, 15, 0))
        self.assertEqual(storm["temp_c"], 26.0)
        self.assertEqual(storm["wind_kt"], 24.0)
        self.assertEqual(storm["visibility_km"], 1.5)
        self.assertEqual(storm["sky_cover"], "SCT")
        self.assertEqual(storm["wxcodes"], "+TSRA")
        self.assertTrue(storm["metar"].startswith("VTBD 290800Z"))
        self.assertEqual(calm["wind_kt"], 0.0)
        self.assertEqual(calm["visibility_km"], 10.0)
        self.assertEqual(calm["wxcodes"], "")

    def test_relative_humidity_matches_iem(self):
        self.assertEqual(relative_humidity(30.0, 20.0), 55.0)
        self.assertEqual(relative_humidity(26.0, 23.0), 84.0)
        self.assertIsNone(relative_humidity(30.0, None))


class WeatherFlagsTest(unittest.TestCase):
    def test_flags_follow_metar_weather_codes(self):
        flags = weather_flags(pd.Series(["-RA", "+TSRA", "VCSH", "", None]))
        self.assertEqual(flags["is_rain"].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(flags["is_heavy_rain"].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(flags["is_thunderstorm"].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(flags["is_shower"].tolist(), [0, 0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
