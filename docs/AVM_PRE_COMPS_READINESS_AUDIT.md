# S.P.E.C. NYC Pre-Comps Readiness Audit

Audit date: 2026-05-05

This audit defines the DS controls that must be in place before implementing the comparable-sales engine.

## Readiness Decision

Status: ready to start comps-engine implementation.

The current trainer now blocks the known pre-comps leakage paths:

- Model-critical ETL-imputed sqft/year-built values are reset to missing before model fitting.
- Model imputers are fit inside the training pipeline, not across the full dataset.
- The old `h3_price_lag` target aggregate has been replaced by strict as-of H3 local market features.
- Internal Optuna validation recomputes as-of features from the fit window before scoring the validation window.
- Sale-validity labels are generated before modeling.
- Target-derived fields are blocked from model features and routing.
- Row-level split manifests are materialized for same-row challenger/champion checks.
- Each new model package can carry `pre_comps_readiness.json` and `split_manifest.csv`.

## Implemented Controls

### Sale Validity

Implemented in `src/pre_comps.py`.

Generated fields:

- `sale_validity_status`
- `sale_validity_reasons`

Statuses:

- `valid_training_sale`
- `review`
- `exclude_training`

Current review/exclusion checks:

- missing or below-min sale price
- extreme sale price
- extreme low/high PPSF
- possible unit-identity collision
- rapid resale
- ETL-imputed sqft
- ETL-imputed year built
- missing H3 index
- missing property ID

Only `exclude_training` rows are dropped automatically. `review` rows remain available for model training until a stricter policy is chosen, but the labels are now available for comp eligibility, slice analysis, and high-risk review queues.

### Train-Fit Imputation

Implemented in `restore_model_critical_missingness`.

ETL keeps imputed values for data-quality reporting, but the model path now resets these fields to missing when ETL imputed them:

- `gross_square_feet`
- `year_built`
- `building_age` when `year_built` is reset

The sklearn pipeline then fits imputation on the training split only.

### Strict As-Of Local Market Features

Implemented in `add_asof_local_market_features`.

New features:

- `h3_prior_sale_count`
- `h3_prior_median_price`
- `h3_prior_median_ppsf`

Rules:

- Training rows use expanding historical rows strictly before the row sale date.
- Holdout rows use training split rows only.
- Holdout targets are never used to create holdout features.
- Same-day sales are not used as prior evidence.

### Target-Derived Field Blocking

Current blocked fields include:

- `sale_price`
- `price_tier`
- `price_per_sqft`
- `price_change_pct`
- `previous_sale_price`
- `previous_sale_date`
- `days_since_last_sale`
- `sale_sequence`
- `is_latest_sale`
- prediction/output columns

This preserves a strict boundary before adding historical comps features.

### Row/Split Manifests

Implemented in `build_split_manifest_frame`.

Each row gets a stable `row_id` derived from property/sale identity fields. New packages can include:

- `split_manifest.csv`
- row-count summary
- train row hash
- test row hash
- full row hash
- duplicate row ID count

## Smoke Evidence

Targeted tests:

```bash
python3 -m unittest tests.test_pre_comps tests.test_model_feature_contracts tests.test_model_temporal tests.test_package_builder tests.test_inference -v
```

Result: 21 tests passed.

Package smoke:

```bash
python3 -m src.mlops.artifact_contract --package-dir models/packages/spec_nyc_avm_v2_20260505T045612Z_b6538c8 --allow-pending-release --min-train-rows 1
```

Result: model package contract passed.

Smoke package generated:

- `models/packages/spec_nyc_avm_v2_20260505T045612Z_b6538c8/pre_comps_readiness.json`
- `models/packages/spec_nyc_avm_v2_20260505T045612Z_b6538c8/split_manifest.csv`

## Work Completed After Readiness

- Added the as-of comparable-sales engine in `src/features/comps.py`.
- Added comparable-sales aggregate features to the model feature contract.
- Added package evidence artifacts for selected comps and high-error review samples.

## Remaining Work After Comps Engine

These are not blockers to starting the comps engine, but they remain important:

- Add target modes: raw price, log price, and log PPSF.
- Add rolling-origin validation folds.
- Add calibrated intervals, confidence, hit/no-hit, and abstention.
- Add fairness/proxy gates.
- Add random review sample artifacts.
