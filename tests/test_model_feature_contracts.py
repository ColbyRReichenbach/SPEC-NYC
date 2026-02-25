import unittest

import numpy as np
import pandas as pd

from src.model import (
    build_feature_drift_by_segment_time,
    build_feature_missingness_by_segment_time,
    validate_training_feature_contract,
)


class TestModelFeatureContracts(unittest.TestCase):
    def test_validate_training_feature_contract_blocks_target_derived_feature(self):
        train_df = pd.DataFrame({"sale_price": [500000.0, 600000.0]})
        test_df = pd.DataFrame({"sale_price": [550000.0]})
        with self.assertRaises(ValueError):
            validate_training_feature_contract(
                train_df,
                test_df,
                feature_columns=["sale_price"],
                router_columns=[],
            )

    def test_validate_training_feature_contract_blocks_undocumented_feature(self):
        train_df = pd.DataFrame({"mystery_feature": [1.0, 2.0]})
        test_df = pd.DataFrame({"mystery_feature": [1.5]})
        with self.assertRaises(ValueError):
            validate_training_feature_contract(
                train_df,
                test_df,
                feature_columns=["mystery_feature"],
                router_columns=[],
            )

    def test_build_feature_missingness_by_segment_time(self):
        df = pd.DataFrame(
            {
                "sale_date": pd.to_datetime(["2024-01-10", "2024-01-20", "2024-04-15", "2024-04-20"]),
                "property_segment": ["ELEVATOR", "ELEVATOR", "WALKUP", "WALKUP"],
                "gross_square_feet": [900.0, np.nan, 850.0, np.nan],
                "borough": ["1", "1", "3", "3"],
            }
        )
        out = build_feature_missingness_by_segment_time(
            df,
            feature_columns=["gross_square_feet", "borough"],
        )
        self.assertFalse(out.empty)
        self.assertIn("segment", out.columns)
        self.assertIn("period", out.columns)
        self.assertIn("missing_rate", out.columns)
        elev_rows = out[(out["segment"] == "ELEVATOR") & (out["feature"] == "gross_square_feet")]
        self.assertTrue((elev_rows["missing_rate"] == 0.5).any())

    def test_build_feature_drift_by_segment_time_flags_shift(self):
        train_rows = 80
        test_rows = 40
        train_df = pd.DataFrame(
            {
                "sale_date": pd.date_range("2023-01-01", periods=train_rows, freq="D"),
                "property_segment": ["ELEVATOR"] * train_rows,
                "gross_square_feet": np.random.default_rng(7).normal(1000, 40, train_rows),
                "borough": ["1"] * train_rows,
            }
        )
        test_df = pd.DataFrame(
            {
                "sale_date": pd.date_range("2024-01-01", periods=test_rows, freq="D"),
                "property_segment": ["ELEVATOR"] * test_rows,
                "gross_square_feet": np.random.default_rng(11).normal(5000, 30, test_rows),
                "borough": ["1"] * test_rows,
            }
        )
        out = build_feature_drift_by_segment_time(
            train_df,
            test_df,
            feature_columns=["gross_square_feet", "borough"],
        )
        self.assertFalse(out.empty)
        numeric = out[out["feature"] == "gross_square_feet"]
        self.assertTrue((numeric["status"] == "alert").any())


if __name__ == "__main__":
    unittest.main()
