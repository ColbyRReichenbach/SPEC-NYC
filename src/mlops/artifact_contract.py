"""Audit-grade model package contract validation for S.P.E.C. NYC."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.inference import (
    FORBIDDEN_TARGET_DERIVED_COLUMNS,
    validate_feature_columns_for_inference,
    validate_router_columns_for_inference,
)


REQUIRED_PACKAGE_FILES = [
    "model.joblib",
    "metrics.json",
    "model_card.md",
    "training_manifest.json",
    "data_manifest.json",
    "feature_contract.json",
    "validation_report.json",
    "slice_scorecard.csv",
    "temporal_scorecard.csv",
    "drift_report.json",
    "explainability_manifest.json",
    "release_decision.json",
    "artifact_hashes.json",
]

HASHED_REQUIRED_FILES = [name for name in REQUIRED_PACKAGE_FILES if name != "artifact_hashes.json"]

REQUIRED_METRICS_OVERALL_FIELDS = ["n", "ppe10", "mdape"]
REQUIRED_METRICS_METADATA_FIELDS = [
    "model_package_id",
    "model_version",
    "dataset_version",
    "feature_contract_version",
    "train_rows",
    "test_rows",
    "feature_columns",
    "target",
    "target_transform",
    "trained_at_utc",
]
REQUIRED_FEATURE_CONTRACT_FIELDS = ["feature_contract_version", "features"]
REQUIRED_FEATURE_FIELDS = [
    "name",
    "dtype",
    "source",
    "owner",
    "description",
    "null_policy",
    "inference_available",
    "point_in_time_available",
]
REQUIRED_ROUTER_FIELD_FIELDS = REQUIRED_FEATURE_FIELDS
REQUIRED_TRAINING_MANIFEST_FIELDS = [
    "model_package_id",
    "command",
    "git_sha",
    "python_version",
    "package_versions",
    "random_seed",
    "train_test_split",
    "model_class",
    "hyperparameters",
    "target",
    "target_transform",
    "preprocessing_steps",
    "optimization_objective",
    "run_started_at_utc",
    "run_finished_at_utc",
]
REQUIRED_DATA_MANIFEST_FIELDS = [
    "dataset_version",
    "sources",
    "raw_row_count",
    "post_filter_row_count",
    "schema_hash",
    "data_snapshot_sha256",
    "min_sale_date",
    "max_sale_date",
    "created_at_utc",
    "known_limitations",
]
REQUIRED_DATA_SOURCE_FIELDS = ["name", "uri", "extracted_at_utc", "row_count"]
REQUIRED_VALIDATION_REPORT_FIELDS = [
    "model_package_id",
    "gate_results",
    "overall_metrics",
    "slice_metrics",
    "temporal_metrics",
    "confidence_metrics",
    "fairness_proxy_metrics",
    "known_failures",
    "validation_status",
]
REQUIRED_EXPLAINABILITY_FIELDS = [
    "model_package_id",
    "global_explainability_artifacts",
    "local_explainability_method",
    "feature_importance_artifact",
    "limitations",
]
REQUIRED_RELEASE_DECISION_FIELDS = [
    "proposal_id",
    "decision",
    "candidate_package_id",
    "previous_champion_package_id",
    "rollback_package_id",
    "approver",
    "reason",
    "decided_at_utc",
    "artifact_hashes_sha256",
]
REQUIRED_MODEL_CARD_SECTIONS = [
    "## Intended Use",
    "## Prohibited Use",
    "## Data Sources",
    "## Training Window",
    "## Validation Window",
    "## Model Type",
    "## Target",
    "## Features",
    "## Leakage Controls",
    "## Performance",
    "## Slice Performance",
    "## Confidence and Intervals",
    "## Fairness and Proxy Audit",
    "## Limitations",
    "## Known Failure Modes",
    "## Monitoring Plan",
    "## Rollback Plan",
]


@dataclass
class ContractViolation:
    check: str
    message: str
    path: str | None = None


@dataclass
class ModelPackageValidationResult:
    passed: bool
    package_dir: str
    violations: list[ContractViolation]
    artifacts: list[str]

    def format(self) -> str:
        if self.passed:
            return f"model package contract passed: {self.package_dir}"
        lines = [f"model package contract failed: {self.package_dir}"]
        for violation in self.violations:
            location = f" ({violation.path})" if violation.path else ""
            lines.append(f"- [{violation.check}] {violation.message}{location}")
        return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_hashes(package_dir: Path, files: Iterable[str] = HASHED_REQUIRED_FILES) -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    for rel_path in files:
        path = package_dir / rel_path
        if path.exists() and path.is_file():
            file_hashes[rel_path] = sha256_file(path)
    return {"algorithm": "sha256", "files": file_hashes}


def validate_model_package(
    package_dir: Path,
    *,
    min_train_rows: int = 5000,
    require_approved_release: bool = True,
) -> ModelPackageValidationResult:
    violations: list[ContractViolation] = []
    artifacts: list[str] = []

    if not package_dir.exists():
        return ModelPackageValidationResult(
            passed=False,
            package_dir=str(package_dir),
            violations=[
                ContractViolation(
                    check="package_dir",
                    message="model package directory does not exist",
                    path=str(package_dir),
                )
            ],
            artifacts=[],
        )
    if not package_dir.is_dir():
        return ModelPackageValidationResult(
            passed=False,
            package_dir=str(package_dir),
            violations=[
                ContractViolation(
                    check="package_dir",
                    message="model package path is not a directory",
                    path=str(package_dir),
                )
            ],
            artifacts=[],
        )

    if (package_dir / "legacy_artifact.json").exists():
        violations.append(
            ContractViolation(
                check="legacy_marker",
                message="legacy model packages are not production eligible",
                path=str(package_dir / "legacy_artifact.json"),
            )
        )

    for rel_path in REQUIRED_PACKAGE_FILES:
        path = package_dir / rel_path
        if path.exists() and path.is_file():
            artifacts.append(str(path))
        else:
            violations.append(
                ContractViolation(
                    check="required_file",
                    message=f"missing required file: {rel_path}",
                    path=str(path),
                )
            )

    payloads: dict[str, Any] = {}
    for rel_path in [
        "metrics.json",
        "training_manifest.json",
        "data_manifest.json",
        "feature_contract.json",
        "validation_report.json",
        "explainability_manifest.json",
        "release_decision.json",
        "artifact_hashes.json",
    ]:
        path = package_dir / rel_path
        if not path.exists():
            continue
        try:
            payloads[rel_path] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            violations.append(
                ContractViolation(
                    check="json_parse",
                    message=f"failed to parse {rel_path}: {exc}",
                    path=str(path),
                )
            )

    metrics = _dict_payload(payloads.get("metrics.json"))
    feature_contract = _dict_payload(payloads.get("feature_contract.json"))
    training_manifest = _dict_payload(payloads.get("training_manifest.json"))
    data_manifest = _dict_payload(payloads.get("data_manifest.json"))
    validation_report = _dict_payload(payloads.get("validation_report.json"))
    explainability = _dict_payload(payloads.get("explainability_manifest.json"))
    release_decision = _dict_payload(payloads.get("release_decision.json"))
    artifact_hashes = _dict_payload(payloads.get("artifact_hashes.json"))

    _require_fields(metrics.get("overall"), REQUIRED_METRICS_OVERALL_FIELDS, "metrics.overall", "metrics.json", violations)
    metadata = metrics.get("metadata") if isinstance(metrics.get("metadata"), dict) else {}
    _require_fields(metadata, REQUIRED_METRICS_METADATA_FIELDS, "metrics.metadata", "metrics.json", violations)

    _require_fields(feature_contract, REQUIRED_FEATURE_CONTRACT_FIELDS, "feature_contract", "feature_contract.json", violations)
    feature_names = _validate_feature_declarations(
        feature_contract.get("features"),
        violations,
        check_prefix="feature_contract.feature",
        required_fields=REQUIRED_FEATURE_FIELDS,
        require_non_empty=True,
    )
    router_contract_names = _validate_router_contract(feature_contract, violations)

    metric_features = metadata.get("feature_columns", [])
    if not isinstance(metric_features, list) or not metric_features:
        violations.append(
            ContractViolation(
                check="metrics_feature_columns",
                message="metrics.metadata.feature_columns must be a non-empty list",
                path="metrics.json",
            )
        )
    else:
        normalized_metric_features = [str(feature) for feature in metric_features]
        try:
            validate_feature_columns_for_inference(
                normalized_metric_features,
                context="metrics.metadata.feature_columns",
            )
        except ValueError as exc:
            violations.append(
                ContractViolation(
                    check="feature_columns",
                    message=str(exc),
                    path="metrics.json",
                )
            )
        forbidden = sorted(set(_lowered(normalized_metric_features)) & FORBIDDEN_TARGET_DERIVED_COLUMNS)
        if forbidden:
            violations.append(
                ContractViolation(
                    check="forbidden_feature",
                    message=f"metrics metadata includes target-derived features: {forbidden}",
                    path="metrics.json",
                )
            )
        if feature_names and sorted(normalized_metric_features) != sorted(feature_names):
            violations.append(
                ContractViolation(
                    check="feature_contract_mismatch",
                    message="metrics feature_columns must exactly match feature_contract feature names",
                    path="metrics.json",
                )
            )

    _validate_router_metadata(metadata, router_contract_names, violations)

    try:
        train_rows = int(metadata.get("train_rows", 0))
        if train_rows < min_train_rows:
            violations.append(
                ContractViolation(
                    check="train_rows",
                    message=f"train_rows={train_rows} below production threshold {min_train_rows}",
                    path="metrics.json",
                )
            )
    except Exception:
        violations.append(
            ContractViolation(
                check="train_rows",
                message="metrics.metadata.train_rows must be an integer",
                path="metrics.json",
            )
        )

    _require_fields(training_manifest, REQUIRED_TRAINING_MANIFEST_FIELDS, "training_manifest", "training_manifest.json", violations)
    _require_fields(data_manifest, REQUIRED_DATA_MANIFEST_FIELDS, "data_manifest", "data_manifest.json", violations)
    _validate_data_sources(data_manifest, violations)
    _require_fields(validation_report, REQUIRED_VALIDATION_REPORT_FIELDS, "validation_report", "validation_report.json", violations)
    _require_fields(explainability, REQUIRED_EXPLAINABILITY_FIELDS, "explainability_manifest", "explainability_manifest.json", violations)
    _require_fields(release_decision, REQUIRED_RELEASE_DECISION_FIELDS, "release_decision", "release_decision.json", violations)

    if require_approved_release and release_decision and str(release_decision.get("decision", "")).lower() != "approved":
        violations.append(
            ContractViolation(
                check="release_decision",
                message="release_decision.decision must be approved for production eligibility",
                path="release_decision.json",
            )
        )

    _validate_model_card(package_dir / "model_card.md", violations)
    _validate_artifact_hashes(package_dir, artifact_hashes, violations)

    return ModelPackageValidationResult(
        passed=len(violations) == 0,
        package_dir=str(package_dir),
        violations=violations,
        artifacts=artifacts,
    )


def _dict_payload(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _lowered(values: Iterable[Any]) -> list[str]:
    return [str(value).strip().lower() for value in values]


def _require_fields(
    payload: Any,
    required: list[str],
    check: str,
    path: str,
    violations: list[ContractViolation],
) -> None:
    if not isinstance(payload, dict):
        violations.append(ContractViolation(check=check, message="payload must be an object", path=path))
        return
    missing = [field for field in required if field not in payload or payload.get(field) in (None, "")]
    if missing:
        violations.append(
            ContractViolation(
                check=check,
                message=f"missing required fields: {missing}",
                path=path,
            )
        )


def _validate_feature_declarations(
    raw_features: Any,
    violations: list[ContractViolation],
    *,
    check_prefix: str,
    required_fields: list[str],
    require_non_empty: bool,
) -> list[str]:
    if raw_features is None and not require_non_empty:
        return []
    if not isinstance(raw_features, list) or (require_non_empty and not raw_features):
        violations.append(
            ContractViolation(
                check=f"{check_prefix}s",
                message="must be a non-empty list" if require_non_empty else "must be a list",
                path="feature_contract.json",
            )
        )
        return []

    feature_names: list[str] = []
    seen: set[str] = set()
    for idx, feature in enumerate(raw_features):
        if not isinstance(feature, dict):
            violations.append(
                ContractViolation(
                    check=check_prefix,
                    message=f"entry at index {idx} must be an object",
                    path="feature_contract.json",
                )
            )
            continue
        _require_fields(
            feature,
            required_fields,
            f"{check_prefix}[{idx}]",
            "feature_contract.json",
            violations,
        )
        name = str(feature.get("name", "")).strip()
        if not name:
            continue
        lowered = name.lower()
        feature_names.append(name)
        if lowered in seen:
            violations.append(
                ContractViolation(
                    check=f"{check_prefix}.duplicate",
                    message=f"duplicate name: {name}",
                    path="feature_contract.json",
                )
            )
        seen.add(lowered)
        if lowered in FORBIDDEN_TARGET_DERIVED_COLUMNS:
            violations.append(
                ContractViolation(
                    check="forbidden_feature",
                    message=f"field is target-derived and forbidden: {name}",
                    path="feature_contract.json",
                )
            )
        if feature.get("inference_available") is not True:
            violations.append(
                ContractViolation(
                    check="inference_availability",
                    message=f"field must be inference_available=true: {name}",
                    path="feature_contract.json",
                )
            )
        if feature.get("point_in_time_available") is not True:
            violations.append(
                ContractViolation(
                    check="point_in_time_availability",
                    message=f"field must be point_in_time_available=true: {name}",
                    path="feature_contract.json",
                )
            )
    return feature_names


def _validate_router_contract(feature_contract: dict[str, Any], violations: list[ContractViolation]) -> list[str]:
    router_names = _validate_feature_declarations(
        feature_contract.get("router_columns"),
        violations,
        check_prefix="feature_contract.router_column",
        required_fields=REQUIRED_ROUTER_FIELD_FIELDS,
        require_non_empty=False,
    )
    if router_names:
        try:
            validate_router_columns_for_inference(router_names, context="feature_contract.router_columns")
        except ValueError as exc:
            violations.append(
                ContractViolation(
                    check="router_columns",
                    message=str(exc),
                    path="feature_contract.json",
                )
            )
    return router_names


def _validate_router_metadata(
    metadata: dict[str, Any],
    router_contract_names: list[str],
    violations: list[ContractViolation],
) -> None:
    raw_router_columns = metadata.get("router_columns", [])
    if raw_router_columns in (None, ""):
        raw_router_columns = []
    if not raw_router_columns:
        return
    if not isinstance(raw_router_columns, list):
        violations.append(
            ContractViolation(
                check="metrics_router_columns",
                message="metrics.metadata.router_columns must be a list when present",
                path="metrics.json",
            )
        )
        return

    router_columns = [str(column).strip() for column in raw_router_columns]
    try:
        validate_router_columns_for_inference(router_columns, context="metrics.metadata.router_columns")
    except ValueError as exc:
        violations.append(
            ContractViolation(
                check="router_columns",
                message=str(exc),
                path="metrics.json",
            )
        )
    if router_contract_names and sorted(router_columns) != sorted(router_contract_names):
        violations.append(
            ContractViolation(
                check="router_contract_mismatch",
                message="metrics router_columns must match feature_contract router_columns",
                path="metrics.json",
            )
        )
    if router_columns and not router_contract_names:
        violations.append(
            ContractViolation(
                check="router_contract_missing",
                message="metrics router_columns are declared but feature_contract.router_columns is missing",
                path="feature_contract.json",
            )
        )


def _validate_data_sources(data_manifest: dict[str, Any], violations: list[ContractViolation]) -> None:
    sources = data_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        violations.append(
            ContractViolation(
                check="data_manifest.sources",
                message="sources must be a non-empty list",
                path="data_manifest.json",
            )
        )
        return
    for idx, source in enumerate(sources):
        _require_fields(
            source,
            REQUIRED_DATA_SOURCE_FIELDS,
            f"data_manifest.sources[{idx}]",
            "data_manifest.json",
            violations,
        )


def _validate_model_card(path: Path, violations: list[ContractViolation]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    missing = [section for section in REQUIRED_MODEL_CARD_SECTIONS if section not in text]
    if missing:
        violations.append(
            ContractViolation(
                check="model_card_sections",
                message=f"model_card.md missing required sections: {missing}",
                path=str(path),
            )
        )


def _validate_artifact_hashes(
    package_dir: Path,
    artifact_hashes: dict[str, Any],
    violations: list[ContractViolation],
) -> None:
    if not artifact_hashes:
        return
    if artifact_hashes.get("algorithm") != "sha256":
        violations.append(
            ContractViolation(
                check="artifact_hashes.algorithm",
                message="artifact_hashes.algorithm must be sha256",
                path="artifact_hashes.json",
            )
        )
    files = artifact_hashes.get("files")
    if not isinstance(files, dict):
        violations.append(
            ContractViolation(
                check="artifact_hashes.files",
                message="artifact_hashes.files must be an object",
                path="artifact_hashes.json",
            )
        )
        return

    for rel_path in HASHED_REQUIRED_FILES:
        expected = files.get(rel_path)
        if not expected:
            violations.append(
                ContractViolation(
                    check="artifact_hashes.missing",
                    message=f"missing hash for required file: {rel_path}",
                    path="artifact_hashes.json",
                )
            )
            continue
        path = package_dir / rel_path
        if not path.exists():
            continue
        actual = sha256_file(path)
        if actual != expected:
            violations.append(
                ContractViolation(
                    check="artifact_hashes.mismatch",
                    message=f"hash mismatch for {rel_path}",
                    path=str(path),
                )
            )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Validate an audit-grade S.P.E.C. NYC model package.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument(
        "--allow-pending-release",
        action="store_true",
        help="Validate candidate package structure without requiring release_decision.decision=approved.",
    )
    args = parser.parse_args()

    result = validate_model_package(
        args.package_dir,
        min_train_rows=args.min_train_rows,
        require_approved_release=not args.allow_pending_release,
    )
    print(result.format())
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    _cli()
