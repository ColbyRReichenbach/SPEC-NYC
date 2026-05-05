import unittest

import pandas as pd

from src.features.comps import (
    COMP_FEATURES,
    CompEngineConfig,
    add_asof_comps_features,
    compute_asof_comps_for_frame,
)
from src.inference import validate_feature_columns_for_inference


def _row(
    *,
    property_id: str,
    sale_date: str,
    sale_price: float,
    borough: str = "1",
    segment: str = "WALKUP",
    sqft: float = 1000.0,
    lat: float = 40.75,
    lon: float = -73.98,
    status: str = "valid_training_sale",
) -> dict:
    return {
        "property_id": property_id,
        "bbl": property_id,
        "unit_identifier": "",
        "sale_date": pd.Timestamp(sale_date),
        "sale_price": sale_price,
        "gross_square_feet": sqft,
        "borough": borough,
        "property_segment": segment,
        "building_class": "C1",
        "h3_index": "h3a",
        "latitude": lat,
        "longitude": lon,
        "building_age": 70.0,
        "total_units": 10.0,
        "residential_units": 10.0,
        "sale_validity_status": status,
    }


class TestComparableSalesEngine(unittest.TestCase):
    def test_comps_features_are_inference_contract_safe(self):
        self.assertEqual(validate_feature_columns_for_inference(COMP_FEATURES), COMP_FEATURES)

    def test_asof_comps_never_use_current_same_day_holdout_or_invalid_sales(self):
        train = pd.DataFrame(
            [
                _row(property_id="p1", sale_date="2020-01-01", sale_price=100_000),
                _row(property_id="p2", sale_date="2020-01-15", sale_price=120_000, status="review"),
                _row(property_id="p3", sale_date="2020-02-01", sale_price=130_000),
            ]
        )
        test = pd.DataFrame(
            [
                _row(property_id="t1", sale_date="2020-02-01", sale_price=999_999),
            ]
        )

        result = add_asof_comps_features(
            train,
            test,
            config=CompEngineConfig(top_k=5, min_comps=1, primary_max_age_days=365, fallback_max_age_days=365),
        )

        self.assertEqual(float(result.train_df.loc[0, "comp_count"]), 0.0)
        self.assertEqual(float(result.train_df.loc[2, "comp_count"]), 1.0)
        self.assertEqual(float(result.train_df.loc[2, "comp_median_price"]), 100_000.0)
        self.assertEqual(float(result.test_df.loc[0, "comp_count"]), 1.0)
        self.assertEqual(float(result.test_df.loc[0, "comp_median_price"]), 100_000.0)
        self.assertNotIn(999_999.0, result.selected_comps["comp_sale_price"].tolist())
        self.assertNotIn(120_000.0, result.selected_comps["comp_sale_price"].tolist())
        self.assertNotIn(130_000.0, result.selected_comps["comp_sale_price"].tolist())

    def test_comps_apply_borough_segment_and_similarity_rules(self):
        reference = pd.DataFrame(
            [
                _row(property_id="same", sale_date="2020-01-01", sale_price=100_000, borough="1", segment="WALKUP"),
                _row(property_id="compatible", sale_date="2020-01-02", sale_price=110_000, borough="1", segment="ELEVATOR"),
                _row(property_id="wrong_borough", sale_date="2020-01-03", sale_price=500_000, borough="2", segment="WALKUP"),
                _row(property_id="far", sale_date="2020-01-04", sale_price=900_000, borough="1", segment="WALKUP", lat=41.5),
            ]
        )
        target = pd.DataFrame(
            [
                _row(property_id="target", sale_date="2020-03-01", sale_price=1_000_000, borough="1", segment="WALKUP"),
            ]
        )

        features, selected, manifest = compute_asof_comps_for_frame(
            target,
            reference,
            split_name="test",
            config=CompEngineConfig(top_k=3, min_comps=2, max_distance_km=3.0, fallback_max_age_days=365),
            include_selected=True,
        )

        self.assertEqual(float(features.loc[0, "comp_count"]), 2.0)
        self.assertEqual(set(selected["comp_property_id"]), {"same", "compatible"})
        self.assertEqual(manifest["comp_count"]["no_comp_rate"], 0.0)

    def test_weighted_estimate_uses_target_square_footage_and_comp_ppsf(self):
        reference = pd.DataFrame(
            [
                _row(property_id="p1", sale_date="2020-01-01", sale_price=100_000, sqft=1000),
                _row(property_id="p2", sale_date="2020-02-01", sale_price=240_000, sqft=2000),
            ]
        )
        target = pd.DataFrame(
            [
                _row(property_id="target", sale_date="2020-04-01", sale_price=1_000_000, sqft=1500),
            ]
        )

        features, selected, _ = compute_asof_comps_for_frame(
            target,
            reference,
            split_name="test",
            config=CompEngineConfig(top_k=2, min_comps=1, fallback_max_age_days=365),
            include_selected=True,
        )

        self.assertEqual(float(features.loc[0, "comp_count"]), 2.0)
        self.assertGreater(float(features.loc[0, "comp_weighted_estimate"]), 150_000.0)
        self.assertLess(float(features.loc[0, "comp_weighted_estimate"]), 180_000.0)
        self.assertEqual(selected["comp_rank"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()

