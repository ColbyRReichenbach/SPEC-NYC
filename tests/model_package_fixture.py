import json
from pathlib import Path

from src.mlops.artifact_contract import REQUIRED_MODEL_CARD_SECTIONS, build_artifact_hashes


def write_minimal_model_package(
    package_dir: Path,
    *,
    feature_names: list[str] | None = None,
    router_columns: list[str] | None = None,
    train_rows: int = 12000,
    decision: str = "approved",
) -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    features = feature_names or ["gross_square_feet", "borough", "property_segment"]
    routers = router_columns or []

    (package_dir / "model.joblib").write_text("model-bytes", encoding="utf-8")
    (package_dir / "slice_scorecard.csv").write_text("slice,n,ppe10,mdape\nall,10,0.4,0.2\n", encoding="utf-8")
    (package_dir / "temporal_scorecard.csv").write_text("period,n,ppe10,mdape\n2024Q1,10,0.4,0.2\n", encoding="utf-8")
    (package_dir / "drift_report.json").write_text('{"status":"ok"}', encoding="utf-8")

    metrics = {
        "overall": {"n": 100, "ppe10": 0.4, "mdape": 0.2},
        "metadata": {
            "model_package_id": "spec_nyc_avm_v2_test",
            "model_version": "v2",
            "dataset_version": "fixture_ds",
            "feature_contract_version": "fc_test",
            "train_rows": train_rows,
            "test_rows": 100,
            "feature_columns": features,
            "router_columns": routers,
            "target": "sale_price",
            "target_transform": "none",
            "trained_at_utc": "2026-05-04T00:00:00Z",
        },
    }
    (package_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    feature_contract = {
        "feature_contract_version": "fc_test",
        "features": [
            {
                "name": name,
                "dtype": "float64" if name != "borough" and name != "property_segment" else "string",
                "source": "fixture",
                "owner": "test",
                "description": f"Fixture feature {name}",
                "null_policy": "impute_or_reject",
                "inference_available": True,
                "point_in_time_available": True,
            }
            for name in features
        ],
        "router_columns": [
            {
                "name": name,
                "dtype": "string",
                "source": "fixture",
                "owner": "test",
                "description": f"Fixture router column {name}",
                "null_policy": "fallback_route",
                "inference_available": True,
                "point_in_time_available": True,
            }
            for name in routers
        ],
    }
    (package_dir / "feature_contract.json").write_text(json.dumps(feature_contract), encoding="utf-8")

    training_manifest = {
        "model_package_id": "spec_nyc_avm_v2_test",
        "command": "pytest fixture",
        "git_sha": "testsha",
        "python_version": "3.10",
        "package_versions": {"xgboost": "test"},
        "random_seed": 42,
        "train_test_split": {"type": "time"},
        "model_class": "FixtureRegressor",
        "hyperparameters": {},
        "target": "sale_price",
        "target_transform": "none",
        "preprocessing_steps": ["fixture"],
        "optimization_objective": "mdape",
        "run_started_at_utc": "2026-05-04T00:00:00Z",
        "run_finished_at_utc": "2026-05-04T00:01:00Z",
    }
    (package_dir / "training_manifest.json").write_text(json.dumps(training_manifest), encoding="utf-8")

    data_manifest = {
        "dataset_version": "fixture_ds",
        "sources": [{"name": "fixture", "uri": "memory://fixture", "extracted_at_utc": "2026-05-04T00:00:00Z", "row_count": 12100}],
        "raw_row_count": 12100,
        "post_filter_row_count": 12100,
        "schema_hash": "schemahash",
        "data_snapshot_sha256": "datahash",
        "min_sale_date": "2024-01-01",
        "max_sale_date": "2025-01-01",
        "created_at_utc": "2026-05-04T00:00:00Z",
        "known_limitations": ["fixture"],
    }
    (package_dir / "data_manifest.json").write_text(json.dumps(data_manifest), encoding="utf-8")

    validation_report = {
        "model_package_id": "spec_nyc_avm_v2_test",
        "gate_results": [],
        "overall_metrics": {},
        "slice_metrics": {},
        "temporal_metrics": {},
        "confidence_metrics": {},
        "fairness_proxy_metrics": {},
        "known_failures": [],
        "validation_status": "pass",
    }
    (package_dir / "validation_report.json").write_text(json.dumps(validation_report), encoding="utf-8")

    explainability = {
        "model_package_id": "spec_nyc_avm_v2_test",
        "global_explainability_artifacts": [],
        "local_explainability_method": "fixture",
        "feature_importance_artifact": "fixture",
        "limitations": ["fixture"],
    }
    (package_dir / "explainability_manifest.json").write_text(json.dumps(explainability), encoding="utf-8")

    release_decision = {
        "proposal_id": "proposal_test",
        "decision": decision,
        "candidate_package_id": "spec_nyc_avm_v2_test",
        "previous_champion_package_id": "none",
        "rollback_package_id": "none",
        "approver": "test",
        "reason": "fixture approval",
        "decided_at_utc": "2026-05-04T00:02:00Z",
        "artifact_hashes_sha256": "filled-by-release-system",
    }
    (package_dir / "release_decision.json").write_text(json.dumps(release_decision), encoding="utf-8")

    model_card = "\n\n".join(["# Fixture Model Card", *REQUIRED_MODEL_CARD_SECTIONS])
    (package_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    hashes = build_artifact_hashes(package_dir)
    (package_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return package_dir
