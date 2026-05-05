import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.mlops.artifact_contract import REQUIRED_PACKAGE_FILES, validate_model_package
from src.mlops.package_builder import write_candidate_model_package


class TestPackageBuilder(unittest.TestCase):
    def test_write_candidate_model_package_outputs_contract_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            package_id = "spec_nyc_avm_v2_20260504T000000Z_testsha"
            feature_columns = ["gross_square_feet", "borough"]
            router_columns = ["property_segment"]
            train_df = pd.DataFrame(
                {
                    "sale_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "sale_price": [500000.0, 600000.0],
                    "gross_square_feet": [900.0, 1000.0],
                    "borough": ["1", "1"],
                    "property_segment": ["WALKUP", "WALKUP"],
                }
            )
            test_df = pd.DataFrame(
                {
                    "sale_date": pd.to_datetime(["2024-04-01", "2024-04-02"]),
                    "sale_price": [550000.0, 650000.0],
                    "gross_square_feet": [950.0, 1050.0],
                    "borough": ["1", "1"],
                    "property_segment": ["WALKUP", "WALKUP"],
                }
            )
            metrics = {
                "overall": {"n": 2, "ppe10": 0.5, "mdape": 0.08},
                "per_segment": {"WALKUP": {"n": 2, "ppe10": 0.5, "mdape": 0.08}},
                "per_price_tier": {},
                "metadata": {
                    "model_package_id": package_id,
                    "model_version": "v2",
                    "dataset_version": "fixture_ds",
                    "feature_contract_version": "fc_test",
                    "train_rows": 2,
                    "test_rows": 2,
                    "feature_columns": feature_columns,
                    "router_columns": router_columns,
                    "target": "sale_price",
                    "target_transform": "none",
                    "trained_at_utc": "2026-05-04T00:00:00Z",
                },
            }

            model_path = base / "model.joblib"
            slice_path = base / "slice.csv"
            temporal_path = base / "temporal.csv"
            drift_path = base / "drift.csv"
            comps_manifest_path = base / "comps_manifest.json"
            selected_comps_path = base / "selected_comps.csv"
            high_error_review_path = base / "high_error_review_sample.csv"
            high_error_comps_path = base / "high_error_selected_comps.csv"
            model_path.write_text("model-bytes", encoding="utf-8")
            slice_path.write_text("group_type,group_name,n,ppe10,mdape\nsegment,WALKUP,2,0.5,0.08\n", encoding="utf-8")
            temporal_path.write_text("period,n,ppe10,mdape\n2024Q2,2,0.5,0.08\n", encoding="utf-8")
            drift_path.write_text("feature,status\nborough,ok\ngross_square_feet,warn\n", encoding="utf-8")
            comps_manifest_path.write_text('{"feature_names":["comp_count"]}', encoding="utf-8")
            selected_comps_path.write_text("valuation_row_id,comp_rank,comp_sale_price\nr1,1,500000\n", encoding="utf-8")
            high_error_review_path.write_text("row_id,abs_pct_error,review_status\nr1,0.25,pending\n", encoding="utf-8")
            high_error_comps_path.write_text("valuation_row_id,comp_rank,comp_sale_price\nr1,1,500000\n", encoding="utf-8")

            result = write_candidate_model_package(
                package_dir=base / "package",
                package_id=package_id,
                model_artifact_path=model_path,
                slice_scorecard_path=slice_path,
                temporal_scorecard_path=temporal_path,
                feature_drift_path=drift_path,
                train_df=train_df,
                test_df=test_df,
                raw_row_count=4,
                data_sources=[
                    {
                        "name": "fixture",
                        "uri": "memory://fixture",
                        "extracted_at_utc": "2026-05-04T00:00:00Z",
                        "row_count": 4,
                    }
                ],
                metrics=metrics,
                feature_columns=feature_columns,
                router_columns=router_columns,
                command="pytest fixture",
                git_sha="testsha",
                model_version="v2",
                dataset_version="fixture_ds",
                feature_contract_version="fc_test",
                model_class="FixtureRegressor",
                hyperparameters={},
                random_seed=42,
                train_test_split={"type": "chronological_holdout"},
                target="sale_price",
                target_transform="none",
                preprocessing_steps=["fixture"],
                optimization_objective="mdape",
                run_started_at_utc="2026-05-04T00:00:00Z",
                run_finished_at_utc="2026-05-04T00:01:00Z",
                comps_manifest_path=comps_manifest_path,
                selected_comps_path=selected_comps_path,
                high_error_review_path=high_error_review_path,
                high_error_comps_path=high_error_comps_path,
            )

            for rel_path in REQUIRED_PACKAGE_FILES:
                self.assertTrue((result.package_dir / rel_path).exists(), rel_path)
            self.assertTrue((result.package_dir / "comps_manifest.json").exists())
            self.assertTrue((result.package_dir / "selected_comps.csv").exists())
            self.assertTrue((result.package_dir / "high_error_review_sample.csv").exists())
            self.assertTrue((result.package_dir / "high_error_selected_comps.csv").exists())

            candidate_validation = validate_model_package(
                result.package_dir,
                min_train_rows=1,
                require_approved_release=False,
            )
            self.assertTrue(candidate_validation.passed, candidate_validation.format())

            production_validation = validate_model_package(result.package_dir, min_train_rows=1)
            self.assertFalse(production_validation.passed)
            formatted = production_validation.format()
            self.assertIn("release_decision", formatted)
            self.assertNotIn("required_file", formatted)
            self.assertNotIn("artifact_hashes.mismatch", formatted)


if __name__ == "__main__":
    unittest.main()
