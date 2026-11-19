import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "WebScraping"))

from bot_holidays import CALENDAR_COLUMNS, build_calendar, parse_holidays  # noqa: E402

RENDERED_HTML = """
<div class="holiday-outer-section"><div class="holiday-inner-section">
<div class="section-title"><div class="title"><h3>2569</h3><div class="arrow-next"></div></div></div>
<div class="holiday-list"><div class="holiday-list-container">
<div class="holiday-group false"><div class="month-title"><h3>สิงหาคม</h3></div>
<div class="month-holiday-item"><div class="item-desc"><h3>วันพุธ 12</h3><h3>วันแม่แห่งชาติ</h3></div></div>
</div>
<div class="holiday-group hide"><div class="month-title"><h3>กันยายน</h3></div></div>
<div class="holiday-group false"><div class="month-title"><h3>ตุลาคม</h3></div>
<div class="month-holiday-item"><div class="item-desc"><h3>วันศุกร์ 23</h3><h3>วันปิยมหาราช</h3></div></div>
</div>
</div></div></div></div>
"""


class ParseHolidaysTest(unittest.TestCase):
    def test_reads_month_day_and_name_in_buddhist_year(self):
        h = parse_holidays(RENDERED_HTML)
        self.assertEqual(h["date"].tolist(), [date(2026, 8, 12), date(2026, 10, 23)])
        self.assertEqual(h["holiday_name"].tolist(), ["วันแม่แห่งชาติ", "วันปิยมหาราช"])
        self.assertEqual(h["source"].unique().tolist(), ["bot.or.th"])


class BuildCalendarTest(unittest.TestCase):
    def test_long_weekend_and_neighbouring_workdays(self):
        holidays = pd.DataFrame({"date": [date(2026, 10, 23)], "holiday_name": ["วันปิยมหาราช"]})
        cal = build_calendar(holidays, "2026-10-21", "2026-10-27").set_index("date")
        self.assertEqual(list(cal.reset_index().columns), CALENDAR_COLUMNS)
        self.assertEqual(cal.loc[date(2026, 10, 23), "is_public_holiday"], 1)
        self.assertEqual(cal.loc[[date(2026, 10, 23), date(2026, 10, 24), date(2026, 10, 25)],
                                 "is_long_weekend"].tolist(), [1, 1, 1])
        self.assertEqual(cal.loc[date(2026, 10, 22), "is_workday_before_holiday"], 1)
        self.assertEqual(cal.loc[date(2026, 10, 26), "is_workday_after_holiday"], 1)
        self.assertEqual(cal.loc[date(2026, 10, 21), "is_day_off"], 0)


if __name__ == "__main__":
    unittest.main()
