import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "WebScraping"))

from tmd_weekly_summary import parse_week_list, parse_week_page  # noqa: E402

LIST_HTML = """
<select id="Select">
<option value="/climate/summaryweekly/060920261000">30 สิงหาคม 2569 - 05 กันยายน 2569</option>
<option value="/climate/summaryweekly/310820261200">24 สิงหาคม 2569 - 30 สิงหาคม 2569</option>
<option value="">เลือกช่วงเวลา</option>
</select>
"""

WEEK_HTML = """
<div class="text-start">
<div class="text-title"><h3 id="item-2">ลักษณะอากาศรายภาค</h3><span>30 สิงหาคม 2569 - 05 กันยายน 2569</span></div>
<div><p class="sub-title">กรุงเทพและปริมณฑล</p>
<p class="sub-content"> มีฝนร้อยละ 25-55 ของพื้นที่ ต่อมามีฝนมากกว่าร้อยละ 75 ของพื้นที่
 กับมีฝนหนักบางแห่ง และมีรายงานน้ำท่วมบริเวณเขตบางนา </p>
<table class="table"><thead><tr><td>ปริมาณฝนสูงสุด</td><td>อุณหภูมิสูงสุด</td><td>อุณหภูมิต่ำสุด</td></tr></thead>
<tbody><tr><td>45.2 มม. ที่ โรงเรียนบ้านหนองระแหง\t\tเขตคลองสามวา เมื่อวันที่ 2 ก.ย.</td><td></td><td></td></tr></tbody>
</table></div>
<div><p class="sub-title">ภาคเหนือ</p><p class="sub-content"> มีฝนร้อยละ 40-60 ของพื้นที่ </p>
<table class="table"><tbody><tr><td></td><td></td><td></td></tr></tbody></table></div>
</div>
"""


class WeekListTest(unittest.TestCase):
    def test_reads_week_ranges_and_links(self):
        weeks = parse_week_list(LIST_HTML)
        self.assertEqual(len(weeks), 2)
        self.assertEqual(weeks[1]["week_start"], date(2026, 8, 24))
        self.assertEqual(weeks[1]["week_end"], date(2026, 8, 30))
        self.assertEqual(weeks[1]["url"], "https://www.tmd.go.th/climate/summaryweekly/310820261200")


class WeekPageTest(unittest.TestCase):
    def test_reads_region_summary_and_max_rain_across_months(self):
        week = {"week_start": date(2026, 8, 30), "week_end": date(2026, 9, 5), "url": "u"}
        bkk, north = parse_week_page(WEEK_HTML, week)
        self.assertEqual(bkk["region"], "กรุงเทพและปริมณฑล")
        self.assertEqual((bkk["rain_area_pct_min"], bkk["rain_area_pct_max"]), (25, 75))
        self.assertEqual(bkk["mentions_heavy_rain"], 1)
        self.assertEqual(bkk["mentions_flood"], 1)
        self.assertEqual(bkk["mentions_flash_flood"], 0)
        self.assertEqual(bkk["max_rain_mm"], 45.2)
        self.assertEqual(bkk["max_rain_location"], "โรงเรียนบ้านหนองระแหง เขตคลองสามวา")
        self.assertEqual(bkk["max_rain_date"], date(2026, 9, 2))
        self.assertIsNone(north["max_rain_mm"])


if __name__ == "__main__":
    unittest.main()
