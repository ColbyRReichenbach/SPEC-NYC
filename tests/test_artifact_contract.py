import json
import tempfile
import unittest
from pathlib import Path

from src.mlops.artifact_contract import sha256_file, validate_model_package
from tests.model_package_fixture import write_minimal_model_package


class TestArtifactContract(unittest.TestCase):
    def test_valid_model_package_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = write_minimal_model_package(Path(tmpdir) / "package")
            result = validate_model_package(package_dir, min_train_rows=5000)
            self.assertTrue(result.passed, result.format())

    def test_target_derived_feature_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = write_minimal_model_package(
                Path(tmpdir) / "package",
                feature_names=["gross_square_feet", "price_tier"],
            )
            result = validate_model_package(package_dir, min_train_rows=5000)
            self.assertFalse(result.passed)
            self.assertIn("forbidden_feature", result.format())

    def test_undocumented_inference_feature_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = write_minimal_model_package(
                Path(tmpdir) / "package",
                feature_names=["gross_square_feet", "unapproved_pluto_field"],
            )
            result = validate_model_package(package_dir, min_train_rows=5000)
            self.assertFalse(result.passed)
            self.assertIn("without inference-availability contract", result.format())

    def test_valid_router_columns_pass_when_declared_separately(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = write_minimal_model_package(
                Path(tmpdir) / "package",
                feature_names=["gross_square_feet", "borough"],
                router_columns=["property_segment", "price_tier_proxy"],
            )
            result = validate_model_package(package_dir, min_train_rows=5000)
            self.assertTrue(result.passed, result.format())

    def test_target_derived_router_column_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = write_minimal_model_package(
                Path(tmpdir) / "package",
                router_columns=["property_segment", "price_tier"],
            )
            result = validate_model_package(package_dir, min_train_rows=5000)
            self.assertFalse(result.passed)
            self.assertIn("router_columns", result.format())

    def test_router_metadata_requires_feature_contract_router_declaration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = write_minimal_model_package(Path(tmpdir) / "package", router_columns=["property_segment"])
            contract_path = package_dir / "feature_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract.pop("router_columns")
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            hashes_path = package_dir / "artifact_hashes.json"
            hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            hashes["files"]["feature_contract.json"] = sha256_file(contract_path)
            hashes_path.write_text(json.dumps(hashes), encoding="utf-8")

            result = validate_model_package(package_dir, min_train_rows=5000)
            self.assertFalse(result.passed)
            self.assertIn("router_contract_missing", result.format())

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = write_minimal_model_package(Path(tmpdir) / "package")
            payload = json.loads((package_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
            payload["files"]["metrics.json"] = "0" * 64
            (package_dir / "artifact_hashes.json").write_text(json.dumps(payload), encoding="utf-8")
            result = validate_model_package(package_dir, min_train_rows=5000)
            self.assertFalse(result.passed)
            self.assertIn("artifact_hashes.mismatch", result.format())

    def test_missing_model_card_section_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = write_minimal_model_package(Path(tmpdir) / "package")
            card_path = package_dir / "model_card.md"
            card_path.write_text("# Fixture Model Card\n\n## Intended Use\n", encoding="utf-8")

            hashes_path = package_dir / "artifact_hashes.json"
            hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            hashes["files"]["model_card.md"] = sha256_file(card_path)
            hashes_path.write_text(json.dumps(hashes), encoding="utf-8")

            result = validate_model_package(package_dir, min_train_rows=5000)
            self.assertFalse(result.passed)
            self.assertIn("model_card_sections", result.format())

    def test_sha256_file_is_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "x.txt"
            path.write_text("abc", encoding="utf-8")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
