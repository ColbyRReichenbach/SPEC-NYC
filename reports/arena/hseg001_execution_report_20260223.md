# H-SEG-001 Execution Report (ML + MLOps)

- Date: 2026-02-23
- Hypothesis: `H-SEG-001`
- Owner: `colby`
- Decision: **NO-GO** (do not promote)

## 1) Candidate Training + Lineage
- Candidate run ID: `7f1cff9e88f54fe5a363248cc9a7da7a`
- Registered model: `spec-nyc-avm`
- Candidate model version: `2` (alias `candidate`)
- Strategy: `segmented_router`
- Router mode: `segment_only`
- Hypothesis metadata:
  - `hypothesis_id=H-SEG-001`
  - `change_type=architecture`
  - `feature_set_version=v3.0_segment_router`
  - `dataset_version=20260223_sales_2019_2025`

## 2) Champion vs Candidate (Headline)
- Champion (`models/metrics_v1.json`):
  - PPE10: `0.3254`
  - MdAPE: `0.1637`
- Candidate (`models/metrics_v1_hseg001_20260223.json`):
  - PPE10: `0.2213`
  - MdAPE: `0.2285`

Arena comparison (`reports/arena/proposal_4f9cb9e44f78.json`):
- weighted_segment_mdape_improvement: `-0.4349`
- overall_ppe10_lift: `-0.1041`
- max_major_segment_ppe10_drop: `0.1825`
- min_major_segment_ppe10: `0.1757`
- drift_alert_delta: `0`
- fairness_alert_delta: `-1`
- gate_pass: `false`
- proposal status: `no_winner`

## 3) Policy Gate Evaluation (config/arena_policy.yaml)
- `weighted_segment_mdape_improvement >= 0.05`: **FAIL** (`-0.4349`)
- `max_major_segment_ppe10_drop <= 0.02`: **FAIL** (`0.1825`)
- `major_segment_ppe10_floor >= 0.24`: **FAIL** (`0.1757`)
- `no_new_drift_alerts`: **PASS** (`delta=0`)
- `no_new_fairness_alerts`: **PASS** (`delta=-1`)

## 4) Release Validation
- Smoke readiness: **PASS**
  - `reports/validation/hseg001_smoke_readiness_20260223.md`
  - `reports/validation/hseg001_smoke_readiness_20260223.json`
- Production readiness: **FAIL**
  - `reports/validation/hseg001_production_readiness_20260223.md`
  - `reports/validation/hseg001_production_readiness_20260223.json`
  - Blocking check: `arena_governance_production` (`latest proposal is not approved (status=no_winner)`)

## 5) Recommendation
- **NO-GO for promotion.**
- Rationale:
  - Candidate fails core arena quality gates relative to champion.
  - Production release gating remains blocked by arena governance as expected.

## 6) Command Log
1. `./scripts/ds_workflow.sh arena-status`
2. `./scripts/ds_workflow.sh daily`
3. `./scripts/ds_workflow.sh train-candidate --hypothesis-id H-SEG-001 --change-type architecture --change-summary "Segment-only router using property_segment with global fallback" --owner colby --feature-set-version v3.0_segment_router --dataset-version 20260223_sales_2019_2025 --strategy segmented_router --router-mode segment_only --min-segment-rows 2000 --artifact-tag hseg001_20260223 --run-name hseg001-segment-router-20260223` (failed first attempt: DB not running)
4. `docker compose up -d db && python3 -m src.database create && python3 -m src.etl --input data/raw/annualized_sales_2019_2025.csv --replace-sales --write-report --report-tag hseg001_20260223`
5. `./scripts/ds_workflow.sh train-candidate --hypothesis-id H-SEG-001 --change-type architecture --change-summary "Segment-only router using property_segment with global fallback" --owner colby --feature-set-version v3.0_segment_router --dataset-version 20260223_sales_2019_2025 --strategy segmented_router --router-mode segment_only --min-segment-rows 2000 --artifact-tag hseg001_20260223 --run-name hseg001-segment-router-20260223` (successful)
6. `./scripts/ds_workflow.sh arena-propose`
7. `./scripts/ds_workflow.sh arena-status`
8. `python3 -m src.validate_release --mode smoke --contract-profile canonical --output-md reports/validation/hseg001_smoke_readiness_20260223.md --output-json reports/validation/hseg001_smoke_readiness_20260223.json`
9. `python3 -m src.validate_release --mode production --contract-profile canonical --output-md reports/validation/hseg001_production_readiness_20260223.md --output-json reports/validation/hseg001_production_readiness_20260223.json`

## 7) Artifact Paths
- Candidate run card: `reports/arena/run_card_7f1cff9e88f54fe5a363248cc9a7da7a.md`
- Candidate model: `models/model_v1_hseg001_20260223.joblib`
- Candidate metrics: `models/metrics_v1_hseg001_20260223.json`
- Candidate scorecard: `reports/model/segment_scorecard_v1_hseg001_20260223.csv`
- Candidate predictions: `reports/model/evaluation_predictions_v1_hseg001_20260223.csv`
- Candidate SHAP summary: `reports/model/shap_summary_v1_hseg001_20260223.png`
- Candidate SHAP waterfall: `reports/model/shap_waterfall_v1_hseg001_20260223.png`
- ETL evidence for training snapshot: `reports/data/etl_run_20260223_hseg001_20260223.md`
- ETL stage CSV: `reports/data/etl_run_20260223_hseg001_20260223.csv`
- Arena proposal JSON: `reports/arena/proposal_4f9cb9e44f78.json`
- Arena proposal markdown: `reports/arena/proposal_4f9cb9e44f78.md`
- Arena comparison CSV: `reports/arena/comparison_20260223T134659Z.csv`
- Smoke validation report: `reports/validation/hseg001_smoke_readiness_20260223.md`
- Smoke validation JSON: `reports/validation/hseg001_smoke_readiness_20260223.json`
- Production validation report: `reports/validation/hseg001_production_readiness_20260223.md`
- Production validation JSON: `reports/validation/hseg001_production_readiness_20260223.json`
