import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "WebScraping"))

import external_lakehouse as ext  # noqa: E402


class BronzeSilverTest(unittest.TestCase):
    def setUp(self):
        from pyducklake import Catalog

        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "data").mkdir()
        self.raw = self.tmp / "raw"
        (self.raw / "sun_times").mkdir(parents=True)
        self.catalog = Catalog("lake", str(self.tmp / "t.ducklake"), data_path=str(self.tmp / "data") + "/")
        self.catalog.set_option("data_inlining_row_limit", "0")
        self.catalog.create_namespace("bronze")
        self.catalog.create_namespace("silver")
        self._raw_dir, self._tables = ext.RAW_DIR, ext.TABLES
        ext.RAW_DIR = self.raw
        ext.TABLES = [t for t in ext.TABLES if t["name"] == "sun_times"]

    def tearDown(self):
        ext.RAW_DIR, ext.TABLES = self._raw_dir, self._tables
        self.catalog.connection.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_csv(self, text):
        (self.raw / "sun_times" / "sun_times_daily.csv").write_text(text, encoding="utf-8-sig")

    def test_silver_keeps_scraped_columns_typed_and_deduplicated(self):
        self.write_csv("date,sunrise,sunset,daylight_hours\n"
                       "2026-08-25,2026-08-25 06:06:00,2026-08-25 18:34:00,12.47\n"
                       "2026-08-25,2026-08-25 06:06:00,2026-08-25 18:34:00,12.47\n"
                       ",2026-08-26 06:06:00,2026-08-26 18:33:00,12.45\n")
        ext.ingest_bronze(self.catalog)
        ext.transform_silver(self.catalog)
        con = self.catalog.connection
        cols = [r[0] for r in con.execute("DESCRIBE lake.silver.sun_times").fetchall()]
        types = [r[1] for r in con.execute("DESCRIBE lake.silver.sun_times").fetchall()]
        self.assertEqual(cols, ["date", "sunrise", "sunset", "daylight_hours"])
        self.assertEqual(types, ["DATE", "TIMESTAMP", "TIMESTAMP", "DOUBLE"])
        self.assertEqual(con.execute("SELECT count(*) FROM lake.silver.sun_times").fetchone()[0], 1)

    def test_changed_file_is_appended_and_silver_uses_latest_version(self):
        self.write_csv("date,sunrise,sunset,daylight_hours\n"
                       "2026-08-25,2026-08-25 06:06:00,2026-08-25 18:34:00,12.47\n"
                       "2026-08-26,2026-08-26 06:06:00,2026-08-26 18:33:00,12.45\n")
        ext.ingest_bronze(self.catalog)
        self.write_csv("date,sunrise,sunset,daylight_hours\n"
                       "2026-08-25,2026-08-25 06:07:00,2026-08-25 18:34:00,12.45\n")
        ext.ingest_bronze(self.catalog)
        ext.transform_silver(self.catalog)
        con = self.catalog.connection
        self.assertEqual(con.execute("SELECT count(*) FROM lake.bronze.sun_times_raw").fetchone()[0], 3)
        rows = con.execute("SELECT strftime(sunrise, '%H:%M') FROM lake.silver.sun_times").fetchall()
        self.assertEqual(rows, [("06:07",)])

    def test_same_file_is_not_ingested_twice(self):
        self.write_csv("date,sunrise,sunset,daylight_hours\n2026-08-25,2026-08-25 06:06:00,2026-08-25 18:34:00,12.47\n")
        ext.ingest_bronze(self.catalog)
        ext.ingest_bronze(self.catalog)
        n = self.catalog.connection.execute("SELECT count(*) FROM lake.bronze.sun_times_raw").fetchone()[0]
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
