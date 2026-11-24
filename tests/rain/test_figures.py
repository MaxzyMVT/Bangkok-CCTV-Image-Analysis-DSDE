import tempfile
import unittest
from pathlib import Path

import pandas as pd

from DataAnalysis.rain.figures import make_all


def write_inputs(out: Path):
    pd.DataFrame({"camera_id": [1, 2, 3, 4], "latitude": [13.70, 13.72, 13.74, 13.76],
                  "longitude": [100.50, 100.52, 100.54, 100.56]}).to_csv(out / "camera.csv", index=False)
    pd.DataFrame({"camera_id": [1, 2, 3, 4], "effect_pct": [-30, 25, 2, -5], "ci_low": [-40, 10, -10, None],
                  "ci_high": [-20, 40, 12, None], "n_wet_hours": [5, 5, 5, 2],
                  "label": ["clears", "denser", "unaffected", "too few wet hours"]}).to_csv(out / "camera_effect.csv", index=False)
    rows = [{"cls": c, "window": w, "effect_pct": -17, "ci_low": -20, "ci_high": -13, "n_cameras": 94, "n_wet_hours": 554}
            for c in ("total", "car", "motorcycle") for w in ("all", "rush", "other")]
    pd.DataFrame(rows).to_csv(out / "city_effect.csv", index=False)
    pd.DataFrame({"cls": ["total", "car", "motorcycle"], "effect_pct_points": [-15, -8, -40], "ci_low": [-20, -12, -50],
                  "ci_high": [-10, -4, -30], "n_hours": [9, 9, 9], "n_pairs": [80, 80, 80]}).to_csv(out / "natural_experiment.csv", index=False)
    pd.DataFrame([{"cls": c, "minute": m, "ratio": 0.6 if m < 0 else 1.1, "n_events": 10}
                  for c in ("motorcycle", "car") for m in range(-60, 120, 10)]).to_csv(out / "rebound.csv", index=False)
    pd.DataFrame({"hod": range(6, 22), "effect_pct": [-40] * 16, "ci_low": [-50] * 16, "ci_high": [-30] * 16,
                  "n_cameras": [20] * 16, "meal": [h in (11, 12, 13, 17, 18, 19, 20) for h in range(6, 22)]}).to_csv(out / "mealtime.csv", index=False)
    pd.DataFrame({"feature": ["near_rail", "near_shelter", "in_flood_area"], "denser_share": [0.6, 0.5, 0.1],
                  "other_share": [0.3, 0.4, 0.0], "diff_points": [30, 10, 10], "ci_low": [5, -20, -5],
                  "ci_high": [55, 40, 30], "n_denser": [10] * 3, "n_other": [60] * 3}).to_csv(out / "denser_share.csv", index=False)
    pd.DataFrame({"station": ["สยาม", "Asok"], "latitude": [13.74, 13.73], "longitude": [100.53, 100.56], "denser_cameras": [2, 1],
                  "moto_lost_per_wet_hour": [4.2, 3.1], "camera_ids": ["2 5", "2"]}).to_csv(out / "policy_stations.csv", index=False)


class MakeAllTest(unittest.TestCase):
    def test_make_all_writes_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, fig = Path(tmp) / "out", Path(tmp) / "fig"
            out.mkdir()
            write_inputs(out)
            rail = pd.DataFrame({"latitude": [13.73], "longitude": [100.53]})
            paths = make_all(out, fig, rail=rail)
            self.assertEqual(len(paths), 7)
            for p in paths:
                self.assertTrue(p.exists(), p)
                self.assertGreater(p.stat().st_size, 1024, p)


if __name__ == "__main__":
    unittest.main()
