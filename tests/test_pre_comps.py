import unittest

import numpy as np
import pandas as pd

from src.inference import validate_feature_columns_for_inference
from src.pre_comps import (
    LOCAL_MARKET_FEATURES,
    add_asof_local_market_features,
    add_sale_validity_labels,
    build_split_manifest_frame,
    restore_model_critical_missingness,
    split_manifest_summary,
)


class TestPreCompsControls(unittest.TestCase):
    def test_restore_model_critical_missingness_resets_etl_imputed_values(self):
        frame = pd.DataFrame(
            {
                "gross_square_feet": [1000.0, 1800.0, 0.0],
                "sqft_imputed": [False, True, False],
                "year_built": [1990.0, 1950.0, 0.0],
                "year_built_imputed": [False, True, False],
                "building_age": [36.0, 76.0, 100.0],
            }
        )

        out, report = restore_model_critical_missingness(frame)

        self.assertFalse(pd.isna(out.loc[0, "gross_square_feet"]))
        self.assertTrue(pd.isna(out.loc[1, "gross_square_feet"]))
        self.assertTrue(pd.isna(out.loc[2, "gross_square_feet"]))
        self.assertTrue(pd.isna(out.loc[1, "year_built"]))
        self.assertTrue(pd.isna(out.loc[2, "year_built"]))
        self.assertTrue(pd.isna(out.loc[1, "building_age"]))
        self.assertEqual(report["fields"]["gross_square_feet"]["etl_imputed_rows"], 1)

    def test_sale_validity_labels_exclude_and_review_risky_sales(self):
        frame = pd.DataFrame(
            {
                "sale_price": [500_000.0, 200_000_000.0, 800_000.0, 900_000.0],
                "gross_square_feet": [1000.0, 2000.0, 10.0, 1100.0],
                "sqft_imputed": [False, False, False, True],
                "year_built_imputed": [False, False, False, False],
                "h3_index": ["a", "a", "a", "b"],
                "property_id": ["p1", "p2", "p3", "p4"],
                "bbl": [1, 2, 3, 4],
                "sale_date": pd.to_datetime(["2024-01-01"] * 4),
            }
        )

        out = add_sale_validity_labels(frame)

        self.assertEqual(out.loc[0, "sale_validity_status"], "valid_training_sale")
        self.assertEqual(out.loc[1, "sale_validity_status"], "exclude_training")
        self.assertIn("extreme_sale_price_exclude", out.loc[1, "sale_validity_reasons"])
        self.assertEqual(out.loc[2, "sale_validity_status"], "review")
        self.assertIn("extreme_high_ppsf_review", out.loc[2, "sale_validity_reasons"])
        self.assertEqual(out.loc[3, "sale_validity_status"], "review")
        self.assertIn("etl_sqft_imputed_review", out.loc[3, "sale_validity_reasons"])

    def test_asof_local_features_never_use_current_or_holdout_targets(self):
        train = pd.DataFrame(
            {
                "sale_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"]),
                "sale_price": [100.0, 300.0, 500.0, 700.0],
                "gross_square_feet": [10.0, 10.0, 10.0, 10.0],
                "h3_index": ["a", "a", "a", "a"],
                "sale_validity_status": ["valid_training_sale"] * 4,
            }
        )
        test = pd.DataFrame(
            {
                "sale_date": pd.to_datetime(["2021-01-01"]),
                "sale_price": [10_000.0],
                "gross_square_feet": [10.0],
                "h3_index": ["a"],
                "sale_validity_status": ["valid_training_sale"],
            }
        )

        result = add_asof_local_market_features(train, test, min_h3_prior_count=1)

        self.assertTrue(pd.isna(result.train_df.loc[0, "h3_prior_median_price"]))
        self.assertEqual(float(result.train_df.loc[1, "h3_prior_median_price"]), 100.0)
        self.assertEqual(float(result.train_df.loc[2, "h3_prior_median_price"]), 100.0)
        self.assertEqual(float(result.train_df.loc[3, "h3_prior_median_price"]), 300.0)
        self.assertEqual(float(result.test_df.loc[0, "h3_prior_median_price"]), 400.0)
        self.assertNotEqual(float(result.test_df.loc[0, "h3_prior_median_price"]), 10_000.0)
        self.assertEqual(set(LOCAL_MARKET_FEATURES).issubset(result.train_df.columns), True)

    def test_split_manifest_materializes_stable_train_and_test_rows(self):
        train = pd.DataFrame(
            {
                "property_id": ["p1", "p2"],
                "bbl": [1, 2],
                "unit_identifier": ["", ""],
                "sale_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "sale_price": [100.0, 200.0],
                "h3_index": ["a", "b"],
            }
        )
        test = pd.DataFrame(
            {
                "property_id": ["p3"],
                "bbl": [3],
                "unit_identifier": [""],
                "sale_date": pd.to_datetime(["2020-01-03"]),
                "sale_price": [300.0],
                "h3_index": ["c"],
            }
        )

        manifest = build_split_manifest_frame(train, test)
        summary = split_manifest_summary(manifest)

        self.assertEqual(summary["train_rows"], 2)
        self.assertEqual(summary["test_rows"], 1)
        self.assertEqual(summary["duplicate_row_ids"], 0)
        self.assertEqual(len(manifest["row_id"].iloc[0]), 64)

    def test_target_derived_pre_comps_fields_are_blocked_as_model_features(self):
        blocked = [
            "price_per_sqft",
            "price_change_pct",
            "previous_sale_price",
            "previous_sale_date",
            "days_since_last_sale",
            "is_latest_sale",
        ]
        for feature in blocked:
            with self.subTest(feature=feature):
                with self.assertRaises(ValueError):
                    validate_feature_columns_for_inference([feature])


if __name__ == "__main__":
    unittest.main()
