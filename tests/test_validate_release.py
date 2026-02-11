import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.validate_release import (
    CheckResult,
    _build_etl_smoke_input,
    _build_training_smoke_input,
    check_artifacts,
    evaluate_gates,
)


class TestValidateReleaseHelpers(unittest.TestCase):
    def test_build_etl_smoke_input_has_required_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "etl_smoke.csv"
            _build_etl_smoke_input(path, rows=25, seed=1)
            self.assertTrue(path.exists())
            frame = pd.read_csv(path)
            required = {"sale_date", "sale_price", "bbl", "borough", "block", "lot", "latitude", "longitude"}
            self.assertTrue(required.issubset(set(frame.columns)))
            self.assertEqual(len(frame), 25)

    def test_build_training_smoke_input_has_required_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_smoke.csv"
            _build_training_smoke_input(path, rows=60, seed=2)
            frame = pd.read_csv(path)
            required = {
                "sale_date",
                "sale_price",
                "h3_index",
                "gross_square_feet",
                "year_built",
                "building_age",
                "residential_units",
                "total_units",
                "distance_to_center_km",
                "borough",
                "building_class",
                "property_segment",
                "price_tier",
                "neighborhood",
            }
            self.assertTrue(required.issubset(set(frame.columns)))
            self.assertEqual(len(frame), 60)

    def test_gate_evaluation_flags_blockers(self):
        checks = [
            CheckResult(name="unit_tests", status="pass", detail="ok"),
            CheckResult(name="etl_smoke", status="pass", detail="ok"),
            CheckResult(name="model_smoke", status="pass", detail="ok"),
            CheckResult(name="evaluate_smoke", status="pass", detail="ok"),
            CheckResult(name="explain_smoke", status="pass", detail="ok"),
            CheckResult(name="artifact_inventory", status="pass", detail="ok"),
            CheckResult(name="streamlit_app_smoke", status="fail", detail="crash"),
            CheckResult(name="mlflow_track_smoke", status="pass", detail="ok"),
            CheckResult(name="drift_monitor_smoke", status="pass", detail="ok"),
            CheckResult(name="performance_monitor_smoke", status="pass", detail="ok"),
            CheckResult(name="retrain_policy_smoke", status="pass", detail="ok"),
        ]
        gates = evaluate_gates(checks)
        self.assertEqual(gates["Gate C (Product)"]["status"], "blocked")
        self.assertEqual(gates["Gate E (Release)"]["status"], "blocked")
        self.assertFalse(gates["Gate E (Release)"]["all_green"])

    def test_check_artifacts_detects_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            existing = base / "exists.txt"
            existing.write_text("ok", encoding="utf-8")
            result = check_artifacts([existing, base / "missing.txt"], pattern_paths=[str(base / "*.csv")])
            self.assertEqual(result.status, "fail")


if __name__ == "__main__":
    unittest.main()
