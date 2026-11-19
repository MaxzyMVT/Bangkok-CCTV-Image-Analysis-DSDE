import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "WebScraping"))

from fm91_news import add_text_features, parse_article  # noqa: E402

ARTICLE_HTML = """
<html><head>
<meta property="og:description" content=""ข่าว" ที่มีเครื่องหมายคำพูดทำให้ attribute ขาด" />
</head><body>
<h1>สวพ.FM91</h1>
<nav><ul><li><a class="dropdown-item" href="https://www.fm91bkk.com/newscategory/1">สังคม</a></li></ul></nav>
<p class=""><a href="https://www.fm91bkk.com/">หน้าแรก</a> &gt;
<a href="https://www.fm91bkk.com/newscategory/1/%E0%B8%AA">สังคม</a> &gt;
<a href="https://www.fm91bkk.com/newscategory/13/%E0%B8%88">ข่าวจราจร</a></p>
<h1>อุบัติเหตุกลางแยกรัชดา ถนนลาดพร้าว</h1>
<p class="mt-3" style="color:#999999">วันที่ 25 สิงหาคม 2569 เวลา 07:06 น.</p>
<div class="news-pra"><p>อุบัติเหตุรถชนกัน ถนนลาดพร้าว ขาเข้า มุ่งหน้าแยกรัชดา</p><p>รถติดสะสม กทม.</p></div>
</body></html>
"""


class ParseArticleTest(unittest.TestCase):
    def test_extracts_fields_from_article_page(self):
        a = parse_article(ARTICLE_HTML, 77850)
        self.assertEqual(a["article_id"], 77850)
        self.assertEqual(a["datetime"], datetime(2026, 8, 25, 7, 6))
        self.assertEqual(a["category"], "ข่าวจราจร")
        self.assertEqual(a["title"], "อุบัติเหตุกลางแยกรัชดา ถนนลาดพร้าว")
        self.assertTrue(a["body"].startswith("อุบัติเหตุรถชนกัน"))
        self.assertEqual(a["url"], "https://www.fm91bkk.com/newsarticle/77850")

    def test_page_without_date_is_skipped(self):
        self.assertIsNone(parse_article("<html><h1>ไม่มีวันที่</h1></html>", 1))

    def test_text_features_use_title_and_body(self):
        df = add_text_features(pd.DataFrame([parse_article(ARTICLE_HTML, 77850)]))
        row = df.iloc[0]
        self.assertEqual(row["is_traffic_category"], 1)
        self.assertEqual(row["mentions_accident"], 1)
        self.assertEqual(row["mentions_congestion"], 1)
        self.assertEqual(row["mentions_flood"], 0)
        self.assertEqual(row["bangkok_related"], 1)
        self.assertIn("ลาดพร้าว", row["roads"])
        self.assertIn("แยกรัชดา", row["intersections"])


if __name__ == "__main__":
    unittest.main()
