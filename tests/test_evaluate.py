import unittest
from pathlib import Path

import pandas as pd

from src.evaluate import _resolve_output_paths, build_segment_scorecard, evaluate_predictions
from src.evaluate_avm import (
    avm_metric_summary,
    coefficient_of_dispersion,
    price_related_bias,
    price_related_differential,
)


class TestEvaluate(unittest.TestCase):
    def test_evaluate_predictions_outputs_expected_keys(self):
        frame = pd.DataFrame(
            {
                "sale_price": [100, 100, 100, 200, 200, 200],
                "predicted_price": [100, 110, 90, 210, 190, 220],
                "property_segment": ["A", "A", "A", "B", "B", "B"],
                "price_tier": ["entry", "entry", "core", "premium", "premium", "luxury"],
            }
        )

        metrics = evaluate_predictions(frame)
        self.assertIn("overall", metrics)
        self.assertIn("per_segment", metrics)
        self.assertIn("per_price_tier", metrics)
        self.assertEqual(metrics["overall"]["n"], 6)
        self.assertTrue(0 <= metrics["overall"]["ppe10"] <= 1)
        self.assertIn("ppe5", metrics["overall"])
        self.assertIn("ppe20", metrics["overall"])
        self.assertIn("median_valuation_ratio", metrics["overall"])
        self.assertIn("coefficient_of_dispersion", metrics["overall"])
        self.assertIn("price_related_differential", metrics["overall"])
        self.assertIn("price_related_bias", metrics["overall"])

    def test_segment_scorecard_has_rows(self):
        frame = pd.DataFrame(
            {
                "sale_price": [100, 100, 100, 200, 200, 200, 300, 300],
                "predicted_price": [102, 98, 95, 210, 195, 198, 310, 290],
                "property_segment": ["A", "A", "A", "B", "B", "B", "C", "C"],
                "price_tier": ["entry", "core", "core", "premium", "premium", "luxury", "luxury", "entry"],
            }
        )
        scorecard = build_segment_scorecard(frame)
        self.assertGreaterEqual(len(scorecard), 2)
        self.assertIn("group_type", scorecard.columns)
        self.assertIn("ppe5", scorecard.columns)
        self.assertIn("ppe10", scorecard.columns)
        self.assertIn("ppe20", scorecard.columns)
        self.assertIn("median_valuation_ratio", scorecard.columns)
        self.assertIn("coefficient_of_dispersion", scorecard.columns)

    def test_avm_metric_summary_computes_ratio_diagnostics(self):
        frame = pd.DataFrame(
            {
                "sale_price": [100.0, 100.0, 200.0, 200.0],
                "predicted_price": [90.0, 110.0, 240.0, 160.0],
            }
        )

        metrics = avm_metric_summary(frame)

        self.assertEqual(metrics["n"], 4)
        self.assertAlmostEqual(metrics["ppe5"], 0.0)
        self.assertAlmostEqual(metrics["ppe10"], 0.5)
        self.assertAlmostEqual(metrics["ppe20"], 1.0)
        self.assertAlmostEqual(metrics["mdape"], 0.15)
        self.assertAlmostEqual(metrics["mape"], 0.15)
        self.assertAlmostEqual(metrics["median_valuation_ratio"], 1.0)
        self.assertAlmostEqual(metrics["mean_valuation_ratio"], 1.0)
        self.assertAlmostEqual(metrics["coefficient_of_dispersion"], 15.0)
        self.assertAlmostEqual(metrics["price_related_differential"], 1.0)
        self.assertAlmostEqual(metrics["overvaluation_rate_10"], 0.25)
        self.assertAlmostEqual(metrics["undervaluation_rate_10"], 0.25)

    def test_avm_metric_summary_tracks_interval_coverage_and_hit_rate(self):
        frame = pd.DataFrame(
            {
                "sale_price": [100.0, 200.0, 300.0],
                "predicted_price": [105.0, 210.0, 250.0],
                "prediction_interval_lower": [95.0, 190.0, 260.0],
                "prediction_interval_upper": [110.0, 220.0, 290.0],
                "hit_status": ["hit", "low_confidence_hit", "no_hit"],
            }
        )

        metrics = avm_metric_summary(frame)

        self.assertEqual(metrics["interval_rows"], 3)
        self.assertAlmostEqual(metrics["interval_coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["hit_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["low_confidence_hit_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["no_hit_rate"], 1 / 3)

    def test_standalone_ratio_diagnostics(self):
        ratios = [0.9, 1.1, 1.2, 0.8]
        self.assertAlmostEqual(coefficient_of_dispersion(ratios), 15.0)
        self.assertAlmostEqual(price_related_differential([100, 100, 200, 200], [90, 110, 240, 160]), 1.0)
        self.assertIsNotNone(price_related_bias([100, 150, 200, 250], [100, 150, 220, 300]))

    def test_resolve_output_paths_prod_defaults(self):
        metrics_path, scorecard_path = _resolve_output_paths(
            model_version="v1",
            artifact_tag="prod",
            output_json=None,
            segment_scorecard_csv=None,
        )
        self.assertEqual(metrics_path, Path("models/metrics_v1.json"))
        self.assertEqual(scorecard_path, Path("reports/model/segment_scorecard_v1.csv"))

    def test_resolve_output_paths_smoke_tag(self):
        metrics_path, scorecard_path = _resolve_output_paths(
            model_version="v1",
            artifact_tag="w6_smoke",
            output_json=None,
            segment_scorecard_csv=None,
        )
        self.assertEqual(metrics_path, Path("models/metrics_v1_w6_smoke.json"))
        self.assertEqual(scorecard_path, Path("reports/model/segment_scorecard_v1_w6_smoke.csv"))


if __name__ == "__main__":
    unittest.main()
