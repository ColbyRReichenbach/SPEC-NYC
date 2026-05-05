# S.P.E.C. NYC Model Artifact Contract

Status: v1 contract for the next clean model package  
Owner: MLOps / Model Risk  
Date: 2026-05-04

## Purpose

This contract defines the minimum evidence bundle required before any S.P.E.C. NYC AVM model can be treated as production-eligible.

The model package must allow a reviewer to answer:

- What data was used?
- What features were used?
- Were the features available at inference time?
- Were the features point-in-time safe?
- What code and parameters produced the model?
- What metrics and slice diagnostics did it achieve?
- What are the known risks and limitations?
- Who approved it?
- How can it be rolled back?

Current `v1` artifacts are legacy evidence only. They are not production-eligible under this contract.

## Package Layout

A production-eligible package is a directory, not a loose set of files.

Recommended path:

```text
models/packages/<model_package_id>/
```

Required files:

```text
model.joblib
metrics.json
model_card.md
training_manifest.json
data_manifest.json
feature_contract.json
validation_report.json
slice_scorecard.csv
temporal_scorecard.csv
drift_report.json
explainability_manifest.json
release_decision.json
artifact_hashes.json
```

## Required JSON Schemas

### `metrics.json`

Required top-level fields:

- `overall`
- `metadata`

Required `overall` fields:

- `n`
- `ppe10`
- `mdape`

Required `metadata` fields:

- `model_package_id`
- `model_version`
- `dataset_version`
- `feature_contract_version`
- `train_rows`
- `test_rows`
- `feature_columns`
- `target`
- `target_transform`
- `trained_at_utc`

Optional `metadata` fields:

- `router_columns`

Rules:

- `feature_columns` must exactly match the feature names in `feature_contract.json`.
- `feature_columns` must not include target-derived or post-outcome fields.
- `feature_columns` must be covered by the repo's inference availability allowlist.
- `router_columns`, when present, must match `feature_contract.router_columns`.
- `router_columns` are validated separately from model feature columns because routing can leak even when model features are clean.
- `train_rows` must meet the configured production threshold.

### `feature_contract.json`

Required top-level fields:

- `feature_contract_version`
- `features`

Optional top-level fields:

- `router_columns`

Each feature must include:

- `name`
- `dtype`
- `source`
- `owner`
- `description`
- `null_policy`
- `inference_available`
- `point_in_time_available`

Each router column must include the same fields as a feature declaration.

Rules:

- `inference_available` must be `true`.
- `point_in_time_available` must be `true`.
- Feature names must be unique.
- Router names must be unique.
- Forbidden target-derived fields are never allowed.
- `price_tier_proxy` is allowed for routing only because it is derived from inference-available non-target inputs.
- `price_tier` is never allowed for routing because it is target-derived from sale price.

Forbidden fields include:

- `sale_price`
- `price_tier`
- `predicted_price`
- `prediction_error`
- `abs_pct_error`
- `sale_price_true`
- `sale_price_pred`
- `target`
- `y`

### `training_manifest.json`

Required fields:

- `model_package_id`
- `command`
- `git_sha`
- `python_version`
- `package_versions`
- `random_seed`
- `train_test_split`
- `model_class`
- `hyperparameters`
- `target`
- `target_transform`
- `preprocessing_steps`
- `optimization_objective`
- `run_started_at_utc`
- `run_finished_at_utc`

### `data_manifest.json`

Required fields:

- `dataset_version`
- `sources`
- `raw_row_count`
- `post_filter_row_count`
- `schema_hash`
- `data_snapshot_sha256`
- `min_sale_date`
- `max_sale_date`
- `created_at_utc`
- `known_limitations`

Each source must include:

- `name`
- `uri`
- `extracted_at_utc`
- `row_count`

### `validation_report.json`

Required fields:

- `model_package_id`
- `gate_results`
- `overall_metrics`
- `slice_metrics`
- `temporal_metrics`
- `confidence_metrics`
- `fairness_proxy_metrics`
- `known_failures`
- `validation_status`

### `explainability_manifest.json`

Required fields:

- `model_package_id`
- `global_explainability_artifacts`
- `local_explainability_method`
- `feature_importance_artifact`
- `limitations`

### `release_decision.json`

Required fields:

- `proposal_id`
- `decision`
- `candidate_package_id`
- `previous_champion_package_id`
- `rollback_package_id`
- `approver`
- `reason`
- `decided_at_utc`
- `artifact_hashes_sha256`

Rules:

- `decision` must be `approved` before production release.
- `rollback_package_id` must be populated.
- `reason` must be populated.

### `artifact_hashes.json`

Required shape:

```json
{
  "algorithm": "sha256",
  "files": {
    "model.joblib": "<sha256>",
    "metrics.json": "<sha256>"
  }
}
```

Rules:

- Every required package file except `artifact_hashes.json` must appear in `files`.
- Every listed hash must match the current file bytes.
- A release validator must fail on missing or mismatched hashes.

## Required Model Card Sections

`model_card.md` must contain these headings:

- `## Intended Use`
- `## Prohibited Use`
- `## Data Sources`
- `## Training Window`
- `## Validation Window`
- `## Model Type`
- `## Target`
- `## Features`
- `## Leakage Controls`
- `## Performance`
- `## Slice Performance`
- `## Confidence and Intervals`
- `## Fairness and Proxy Audit`
- `## Limitations`
- `## Known Failure Modes`
- `## Monitoring Plan`
- `## Rollback Plan`

## Production Eligibility

A package is production-eligible only if:

- All required files exist.
- All required JSON files parse.
- Artifact hashes match.
- Feature contract passes.
- Metrics metadata agrees with feature contract.
- Training and data manifests are complete.
- Model card has all required sections.
- Release decision is approved.
- The package is not marked legacy.

Candidate packages may be validated with pending release approval, but they must not be promoted to
`models/packages/current` until governance updates `release_decision.json` to `approved` and records a
rollback pointer.

## Legacy Artifacts

The existing `v1` loose artifacts are legacy:

- `models/model_v1.joblib`
- `models/metrics_v1.json`
- `reports/model/*_v1.*`
- existing `reports/arena/*`
- existing `reports/monitoring/*`

They can be used for historical comparison, but they must not pass production release validation under this contract.
