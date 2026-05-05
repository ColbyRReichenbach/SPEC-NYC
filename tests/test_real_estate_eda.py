import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.eda.real_estate_eda import (
    build_feature_interaction_signals,
    build_model_error_slices,
    build_segment_region_summary,
    run_eda,
)


class TestRealEstateEda(unittest.TestCase):
    def test_segment_region_summary_groups_by_borough_and_segment(self):
        frame = pd.DataFrame(
            {
                "borough": ["1", "1", "3"],
                "property_segment": ["WALKUP", "WALKUP", "SINGLE_FAMILY"],
                "sale_price": [500000, 600000, 700000],
                "price_per_sqft": [500, 600, 700],
                "gross_square_feet": [1000, 1000, 1000],
                "building_age": [80, 70, 60],
                "distance_to_center_km": [2.0, 2.5, 8.0],
            }
        )

        out = build_segment_region_summary(frame)

        self.assertEqual(len(out), 2)
        walkup = out[out["property_segment"] == "WALKUP"].iloc[0]
        self.assertEqual(int(walkup["n"]), 2)
        self.assertEqual(float(walkup["median_ppsf"]), 550.0)

    def test_feature_interaction_signals_detect_non_flat_relationships(self):
        rows = []
        for i in range(300):
            rows.append(
                {
                    "borough": "1",
                    "property_segment": "WALKUP",
                    "price_per_sqft": 100 + i,
                    "distance_to_center_km": 300 - i,
                    "gross_square_feet": 800 + i,
                    "building_age": 50,
                    "total_units": 10,
                    "residential_units": 10,
                }
            )
        frame = pd.DataFrame(rows)

        out = build_feature_interaction_signals(frame, min_rows=100)

        distance = out[(out["feature"] == "distance_to_center_km") & (out["scope"] == "global")].iloc[0]
        self.assertEqual(distance["direction"], "negative")
        self.assertGreater(float(distance["abs_corr"]), 0.9)

    def test_model_error_slices_include_comp_count_bucket_when_available(self):
        frame = pd.DataFrame(
            {
                "sale_price": [100] * 24,
                "predicted_price": [100, 110, 120, 130, 80, 70] * 4,
                "borough": ["1"] * 24,
                "property_segment": ["WALKUP"] * 24,
                "comp_count": [0] * 6 + [1] * 6 + [3] * 6 + [8] * 6,
                "comp_price_dispersion": [0.1] * 24,
            }
        )

        out = build_model_error_slices(frame)

        self.assertIn("comp_count_bucket", set(out["slice_type"]))
        self.assertTrue((out["n"] >= 5).any())

    def test_run_eda_writes_manifest_report_and_backlog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            sales_path = base / "sales.csv"
            predictions_path = base / "predictions.csv"
            rows = []
            for i in range(60):
                rows.append(
                    {
                        "sale_date": f"2024-01-{(i % 28) + 1:02d}",
                        "sale_price": 500000 + i * 1000,
                        "price_per_sqft": 500 + i,
                        "borough": "1",
                        "neighborhood": "N1",
                        "property_segment": "WALKUP",
                        "building_class": "C1",
                        "gross_square_feet": 1000,
                        "year_built": 1930,
                        "building_age": 94,
                        "total_units": 10,
                        "residential_units": 10,
                        "distance_to_center_km": 3.0,
                        "property_id": f"p{i}",
                    }
                )
            pd.DataFrame(rows).to_csv(sales_path, index=False)
            pd.DataFrame(
                {
                    "sale_price": [100, 100, 100, 100, 100, 100],
                    "predicted_price": [100, 110, 120, 130, 80, 70],
                    "borough": ["1"] * 6,
                    "property_segment": ["WALKUP"] * 6,
                }
            ).to_csv(predictions_path, index=False)

            result = run_eda(
                input_csv=sales_path,
                output_dir=base / "eda",
                tag="fixture",
                predictions_csv=predictions_path,
            )

            self.assertTrue(result.manifest_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.hypothesis_backlog_path.exists())
            self.assertTrue(result.model_error_slices_path and result.model_error_slices_path.exists())


if __name__ == "__main__":
    unittest.main()
