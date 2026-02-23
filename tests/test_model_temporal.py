import unittest

import pandas as pd

from src.model import add_temporal_regime_features, build_temporal_scorecard


class TestModelTemporalFeatures(unittest.TestCase):
    def test_add_temporal_regime_features_creates_expected_columns(self):
        train_df = pd.DataFrame({"sale_date": ["2019-01-01", "2021-06-15", "2024-11-20"]})
        test_df = pd.DataFrame({"sale_date": ["2020-04-10", "2023-02-01"]})
        train_df["sale_date"] = pd.to_datetime(train_df["sale_date"])
        test_df["sale_date"] = pd.to_datetime(test_df["sale_date"])

        train_out, test_out = add_temporal_regime_features(train_df, test_df)
        for frame in (train_out, test_out):
            self.assertIn("days_since_2019_start", frame.columns)
            self.assertIn("month_sin", frame.columns)
            self.assertIn("month_cos", frame.columns)
            self.assertIn("rate_regime_bucket", frame.columns)
            self.assertTrue(frame["month_sin"].notna().all())
            self.assertTrue(frame["month_cos"].notna().all())
            self.assertTrue(frame["rate_regime_bucket"].notna().all())

    def test_build_temporal_scorecard_is_time_aware(self):
        df = pd.DataFrame(
            {
                "sale_date": pd.to_datetime(
                    ["2024-01-10", "2024-02-10", "2024-05-10", "2024-08-10", "2024-11-10", "2025-02-10"]
                ),
                "sale_price": [100, 120, 140, 160, 180, 200],
                "predicted_price": [95, 125, 130, 170, 175, 210],
                "property_segment": ["A"] * 6,
                "price_tier": ["entry"] * 6,
            }
        )
        scorecard = build_temporal_scorecard(df, period_freq="Q")
        self.assertGreaterEqual(len(scorecard), 3)
        self.assertEqual(scorecard["period"].tolist(), sorted(scorecard["period"].tolist()))
        self.assertTrue((scorecard["n"] > 0).all())


if __name__ == "__main__":
    unittest.main()
