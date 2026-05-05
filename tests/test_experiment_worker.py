import tempfile
import unittest
from pathlib import Path

from src.experiments.worker import build_comparison_report, build_split_signature


class TestExperimentWorker(unittest.TestCase):
    def test_comparison_report_passes_only_on_locked_dataset_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            split_signature = build_split_signature(
                dataset_version="fixture_ds",
                data_snapshot_sha256="abc123",
                train_rows=80,
                test_rows=20,
                min_sale_date="2024-01-01",
                max_sale_date="2024-12-31",
            )
            bundle = {
                "run_plan": {"baseline_package_id": "champion_package"},
                "baseline_metrics": {"ppe10": 0.2, "mdape": 0.3, "r2": 0.4},
                "dataset_snapshot": {
                    "data_snapshot_sha256": "abc123",
                    "split_signature_sha256": split_signature,
                },
            }
            package_dir = repo_root / "models" / "packages" / "spec_nyc_avm_fixture"
            package_dir.mkdir(parents=True)
            (package_dir / "metrics.json").write_text(
                """
                {
                  "metadata": {
                    "model_package_id": "spec_nyc_avm_fixture",
                    "dataset_version": "fixture_ds",
                    "train_rows": 80,
                    "test_rows": 20
                  },
                  "overall": {
                    "ppe10": 0.205,
                    "mdape": 0.29,
                    "r2": 0.42
                  }
                }
                """,
                encoding="utf-8",
            )
            (package_dir / "data_manifest.json").write_text(
                """
                {
                  "dataset_version": "fixture_ds",
                  "data_snapshot_sha256": "abc123",
                  "min_sale_date": "2024-01-01",
                  "max_sale_date": "2024-12-31"
                }
                """,
                encoding="utf-8",
            )

            report = build_comparison_report(repo_root, bundle, package_dir)

            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["same_dataset_contract"])
            self.assertLess(report["metric_deltas"]["mdape"], 0)

    def test_comparison_report_fails_on_dataset_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            bundle = {
                "run_plan": {"baseline_package_id": "champion_package"},
                "baseline_metrics": {"ppe10": 0.2, "mdape": 0.3, "r2": 0.4},
                "dataset_snapshot": {
                    "data_snapshot_sha256": "locked_snapshot",
                    "split_signature_sha256": "0" * 64,
                },
            }
            package_dir = repo_root / "models" / "packages" / "spec_nyc_avm_fixture"
            package_dir.mkdir(parents=True)
            (package_dir / "metrics.json").write_text(
                """
                {
                  "metadata": {
                    "model_package_id": "spec_nyc_avm_fixture",
                    "dataset_version": "fixture_ds",
                    "train_rows": 80,
                    "test_rows": 20
                  },
                  "overall": {
                    "ppe10": 0.3,
                    "mdape": 0.2,
                    "r2": 0.6
                  }
                }
                """,
                encoding="utf-8",
            )
            (package_dir / "data_manifest.json").write_text(
                """
                {
                  "dataset_version": "fixture_ds",
                  "data_snapshot_sha256": "different_snapshot",
                  "min_sale_date": "2024-01-01",
                  "max_sale_date": "2024-12-31"
                }
                """,
                encoding="utf-8",
            )

            report = build_comparison_report(repo_root, bundle, package_dir)

            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["same_dataset_contract"])
            self.assertIn("did not match", report["blocking_reason"])


if __name__ == "__main__":
    unittest.main()
