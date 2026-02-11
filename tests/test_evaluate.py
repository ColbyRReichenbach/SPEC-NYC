import unittest

import pandas as pd

from src.evaluate import build_segment_scorecard, evaluate_predictions


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
        self.assertIn("ppe10", scorecard.columns)


if __name__ == "__main__":
    unittest.main()

