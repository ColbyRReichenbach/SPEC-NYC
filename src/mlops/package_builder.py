"""Build audit-grade candidate model packages from training outputs."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference import validate_feature_columns_for_inference, validate_router_columns_for_inference
from src.mlops.artifact_contract import REQUIRED_MODEL_CARD_SECTIONS, build_artifact_hashes


PACKAGE_DEPENDENCIES = ["numpy", "pandas", "scikit-learn", "xgboost", "optuna", "joblib"]

FEATURE_DESCRIPTIONS = {
    "gross_square_feet": "Building gross square footage available before valuation.",
    "year_built": "Property construction year available before valuation.",
    "building_age": "Age of the building at valuation time.",
    "residential_units": "Residential unit count.",
    "total_units": "Total unit count.",
    "distance_to_center_km": "Distance to configured city center proxy.",
    "h3_prior_sale_count": "Count of same-H3 historical sales strictly before the valuation date.",
    "h3_prior_median_price": "Same-H3 historical median sale price computed strictly before the valuation date.",
    "h3_prior_median_ppsf": "Same-H3 historical median price per square foot computed strictly before the valuation date.",
    "comp_count": "Number of selected comparable sales available before valuation.",
    "comp_median_price": "Median sale price across selected as-of comparable sales.",
    "comp_median_ppsf": "Median price per square foot across selected as-of comparable sales.",
    "comp_weighted_estimate": "Rule-based comparable-sales estimate from similarity-weighted comps.",
    "comp_price_dispersion": "Median absolute percentage dispersion among selected comparable prices or PPSF.",
    "comp_nearest_distance_km": "Nearest selected comparable sale distance in kilometers.",
    "comp_median_recency_days": "Median days between valuation date and selected comparable sale dates.",
    "comp_local_momentum": "Recent-vs-older selected-comp PPSF momentum proxy.",
    "days_since_2019_start": "Days elapsed since the temporal baseline date.",
    "month_sin": "Cyclical month sine term derived from sale or valuation date.",
    "month_cos": "Cyclical month cosine term derived from sale or valuation date.",
    "borough": "NYC borough identifier.",
    "building_class": "NYC building class category.",
    "property_segment": "Residential property segment used for slicing and optional routing.",
    "neighborhood": "Neighborhood label.",
    "rate_regime_bucket": "Macro time-regime proxy derived from valuation date.",
    "price_tier_proxy": "Non-target tier proxy derived from inference-available property signals.",
}


@dataclass
class PackageBuildResult:
    package_dir: Path
    package_id: str
    artifact_hashes_path: Path


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dependency in PACKAGE_DEPENDENCIES:
        try:
            versions[dependency] = version(dependency)
        except PackageNotFoundError:
            versions[dependency] = "not_installed"
    return versions


def schema_hash(frame: pd.DataFrame) -> str:
    schema = [{"name": str(col), "dtype": str(dtype)} for col, dtype in frame.dtypes.items()]
    payload = json.dumps(schema, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataframe_snapshot_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    row_hashes = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    digest = hashlib.sha256()
    digest.update(schema_hash(normalized).encode("utf-8"))
    digest.update(row_hashes)
    return digest.hexdigest()


def write_candidate_model_package(
    *,
    package_dir: Path,
    package_id: str,
    model_artifact_path: Path,
    slice_scorecard_path: Path,
    temporal_scorecard_path: Path,
    feature_drift_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_row_count: int,
    data_sources: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_columns: list[str],
    router_columns: list[str],
    command: str,
    git_sha: str,
    model_version: str,
    dataset_version: str,
    feature_contract_version: str,
    model_class: str,
    hyperparameters: dict[str, Any],
    random_seed: int,
    train_test_split: dict[str, Any],
    target: str,
    target_transform: str,
    preprocessing_steps: list[str],
    optimization_objective: str,
    run_started_at_utc: str,
    run_finished_at_utc: str,
    shap_artifacts: dict[str, Any] | None = None,
    known_limitations: list[str] | None = None,
    pre_comps_manifest_path: Path | None = None,
    split_manifest_path: Path | None = None,
    comps_manifest_path: Path | None = None,
    selected_comps_path: Path | None = None,
    high_error_review_path: Path | None = None,
    high_error_comps_path: Path | None = None,
) -> PackageBuildResult:
    validate_feature_columns_for_inference(feature_columns, context="package feature_columns")
    if router_columns:
        validate_router_columns_for_inference(router_columns, context="package router_columns")

    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_artifact_path, package_dir / "model.joblib")
    _write_json(package_dir / "metrics.json", metrics)
    shutil.copy2(slice_scorecard_path, package_dir / "slice_scorecard.csv")
    shutil.copy2(temporal_scorecard_path, package_dir / "temporal_scorecard.csv")
    if pre_comps_manifest_path is not None:
        shutil.copy2(pre_comps_manifest_path, package_dir / "pre_comps_readiness.json")
    if split_manifest_path is not None:
        shutil.copy2(split_manifest_path, package_dir / "split_manifest.csv")
    if comps_manifest_path is not None:
        shutil.copy2(comps_manifest_path, package_dir / "comps_manifest.json")
    if selected_comps_path is not None:
        shutil.copy2(selected_comps_path, package_dir / "selected_comps.csv")
    if high_error_review_path is not None:
        shutil.copy2(high_error_review_path, package_dir / "high_error_review_sample.csv")
    if high_error_comps_path is not None:
        shutil.copy2(high_error_comps_path, package_dir / "high_error_selected_comps.csv")

    combined = pd.concat([train_df, test_df], ignore_index=True)
    limitations = known_limitations or [
        "Candidate package generated from available project data only.",
        "Not approved for production use until release_decision.decision is updated by governance workflow.",
        "Public NYC data may omit condition, renovation quality, listing media, concessions, and private transaction context.",
    ]

    _write_json(
        package_dir / "training_manifest.json",
        {
            "model_package_id": package_id,
            "command": command,
            "git_sha": git_sha,
            "python_version": platform.python_version(),
            "package_versions": package_versions(),
            "random_seed": random_seed,
            "train_test_split": train_test_split,
            "model_class": model_class,
            "hyperparameters": hyperparameters,
            "target": target,
            "target_transform": target_transform,
            "preprocessing_steps": preprocessing_steps,
            "optimization_objective": optimization_objective,
            "run_started_at_utc": run_started_at_utc,
            "run_finished_at_utc": run_finished_at_utc,
        },
    )
    _write_json(
        package_dir / "data_manifest.json",
        {
            "dataset_version": dataset_version,
            "sources": data_sources,
            "raw_row_count": int(raw_row_count),
            "post_filter_row_count": int(len(combined)),
            "schema_hash": schema_hash(combined),
            "data_snapshot_sha256": dataframe_snapshot_hash(combined),
            "min_sale_date": _date_bound(combined, "sale_date", "min"),
            "max_sale_date": _date_bound(combined, "sale_date", "max"),
            "data_freshness": {"max_sale_date": _date_bound(combined, "sale_date", "max")},
            "created_at_utc": run_finished_at_utc,
            "known_limitations": limitations,
        },
    )
    _write_json(
        package_dir / "feature_contract.json",
        {
            "feature_contract_version": feature_contract_version,
            "features": [_feature_declaration(name, combined) for name in feature_columns],
            "router_columns": [_feature_declaration(name, combined, router=True) for name in router_columns],
        },
    )
    _write_json(
        package_dir / "validation_report.json",
        {
            "model_package_id": package_id,
            "gate_results": [
                {
                    "gate": "candidate_generation",
                    "status": "pass",
                    "reason": "required candidate package files generated",
                },
                {
                    "gate": "governance_approval",
                    "status": "pending",
                    "reason": "candidate packages are not production-approved automatically",
                },
            ],
            "overall_metrics": metrics.get("overall", {}),
            "slice_metrics": {
                "per_segment": metrics.get("per_segment", {}),
                "per_price_tier": metrics.get("per_price_tier", {}),
            },
            "temporal_metrics": _csv_records(temporal_scorecard_path),
            "confidence_metrics": {"status": "not_implemented"},
            "fairness_proxy_metrics": {"status": "not_evaluated"},
            "known_failures": [],
            "validation_status": "candidate_pending_approval",
        },
    )
    _write_json(package_dir / "drift_report.json", _drift_summary(feature_drift_path))
    _write_json(
        package_dir / "explainability_manifest.json",
        {
            "model_package_id": package_id,
            "global_explainability_artifacts": _shap_artifact_paths(shap_artifacts or {}),
            "local_explainability_method": "shap_waterfall" if shap_artifacts else "not_generated",
            "feature_importance_artifact": str((shap_artifacts or {}).get("summary_plot_path", "not_generated")),
            "limitations": [
                "SHAP values explain model mechanics, not causal effects.",
                "Local explanations are conditional on available public-data features.",
            ],
        },
    )
    _write_json(
        package_dir / "release_decision.json",
        {
            "proposal_id": f"proposal_{package_id}",
            "decision": "pending",
            "candidate_package_id": package_id,
            "previous_champion_package_id": "not_set",
            "rollback_package_id": "not_set",
            "approver": "not_approved",
            "reason": "candidate generated; governance approval not yet performed",
            "gate_results": [
                {"gate": "candidate_generation", "status": "pass"},
                {"gate": "governance_approval", "status": "pending"},
            ],
            "decided_at_utc": run_finished_at_utc,
            "artifact_hashes_sha256": "pending_until_release_approval",
        },
    )
    (package_dir / "model_card.md").write_text(
        _model_card(
            package_id=package_id,
            model_version=model_version,
            dataset_version=dataset_version,
            metrics=metrics,
            feature_columns=feature_columns,
            router_columns=router_columns,
            limitations=limitations,
            run_started_at_utc=run_started_at_utc,
            run_finished_at_utc=run_finished_at_utc,
            target=target,
            target_transform=target_transform,
            model_class=model_class,
        ),
        encoding="utf-8",
    )

    hashes = build_artifact_hashes(package_dir)
    _write_json(package_dir / "artifact_hashes.json", hashes)
    return PackageBuildResult(
        package_dir=package_dir,
        package_id=package_id,
        artifact_hashes_path=package_dir / "artifact_hashes.json",
    )


def _feature_declaration(name: str, frame: pd.DataFrame, *, router: bool = False) -> dict[str, Any]:
    dtype = str(frame[name].dtype) if name in frame.columns else "unknown"
    return {
        "name": name,
        "dtype": dtype,
        "source": "training_feature_engineering",
        "owner": "mlops",
        "description": FEATURE_DESCRIPTIONS.get(name, f"Inference-available AVM field: {name}."),
        "null_policy": "fallback_route" if router else "pipeline_imputation_or_reject",
        "inference_available": True,
        "point_in_time_available": True,
    }


def _model_card(
    *,
    package_id: str,
    model_version: str,
    dataset_version: str,
    metrics: dict[str, Any],
    feature_columns: list[str],
    router_columns: list[str],
    limitations: list[str],
    run_started_at_utc: str,
    run_finished_at_utc: str,
    target: str,
    target_transform: str,
    model_class: str,
) -> str:
    overall = metrics.get("overall", {})
    body = {
        "## Intended Use": "Candidate NYC borough-level AVM research model for governed valuation workflow demonstration.",
        "## Prohibited Use": "Not an appraisal, lending decision, tax assessment, insurance decision, or consumer-facing valuation product.",
        "## Data Sources": f"Dataset version: `{dataset_version}`.",
        "## Training Window": f"Run started at `{run_started_at_utc}`.",
        "## Validation Window": f"Run finished at `{run_finished_at_utc}` with chronological holdout validation.",
        "## Model Type": model_class,
        "## Target": f"`{target}` with target transform `{target_transform}`.",
        "## Features": f"Model features: `{', '.join(feature_columns)}`. Router columns: `{', '.join(router_columns) or 'none'}`.",
        "## Leakage Controls": "Target-derived feature and router columns are blocked by package validation.",
        "## Performance": f"Overall metrics: `{json.dumps(overall, sort_keys=True)}`.",
        "## Slice Performance": "See `slice_scorecard.csv` and `validation_report.json`.",
        "## Confidence and Intervals": "Candidate package does not yet contain calibrated conformal intervals.",
        "## Fairness and Proxy Audit": "Proxy audit is pending; review geography, value band, and segment valuation-ratio gaps before promotion.",
        "## Limitations": " ".join(limitations),
        "## Known Failure Modes": "Sparse local comps, unusual condition, major renovations, non-market sales, and missing public-record fields.",
        "## Monitoring Plan": "Monitor feature drift, prediction drift, hit rate, interval coverage, stale data, and segment decay.",
        "## Rollback Plan": "Use the previous approved champion package once release governance records a rollback pointer.",
    }
    sections = ["# S.P.E.C. NYC AVM Candidate Model Card", f"Package ID: `{package_id}`", f"Model version: `{model_version}`"]
    for section in REQUIRED_MODEL_CARD_SECTIONS:
        sections.extend([section, body[section]])
    return "\n\n".join(sections) + "\n"


def _drift_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_generated", "source_csv": str(path)}
    frame = pd.read_csv(path)
    status_counts = frame["status"].value_counts(dropna=False).to_dict() if "status" in frame.columns else {}
    return {
        "status": "generated",
        "source_csv": str(path),
        "row_count": int(len(frame)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }


def _csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _shap_artifact_paths(shap_artifacts: dict[str, Any]) -> list[str]:
    paths = []
    for key in ["summary_plot_path", "waterfall_plot_path"]:
        value = shap_artifacts.get(key)
        if value:
            paths.append(str(value))
    return paths


def _date_bound(frame: pd.DataFrame, column: str, bound: str) -> str:
    if column not in frame.columns:
        return "unknown"
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        return "unknown"
    value = values.min() if bound == "min" else values.max()
    return str(value.date())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    if pd.isna(value):
        return None
    return value
