# H-SEASON-001 Execution Report (ML Engineer)

- Date: 2026-02-23
- Hypothesis: `H-SEASON-001`
- Decision: **NO-GO** (no promotion)
- Next branch recommendation: **rollback**

## 1) What Was Implemented
Seasonality/regime-safe features added from inference-time signals only:
- `days_since_2019_start` (trend)
- `month_sin`, `month_cos` (cyclical month encoding)
- `rate_regime_bucket` (date-bucketed regime proxy)

Time-aware evaluation design (no random-only validation):
- Chronological sort by `sale_date` + holdout via `time_split(...)` in `src/model.py`.
- Optuna inner split also chronological (`fit` older slice, `valid` newer slice).
- Temporal diagnostics emitted to `reports/model/temporal_scorecard_v1_hseason001_20260223.csv`.

## 2) Candidate Run and Arena Tracking
- Run ID: `6612686186654a3685df7ffca7ca3bde`
- Run card: `reports/arena/run_card_6612686186654a3685df7ffca7ca3bde.md`
- Registered model version: `4`
- Alias: `candidate`
- Tags:
  - `dataset_version=20260223_sales_2019_2025_seasonv1`
  - `feature_set_version=v3.2_seasonality`

## 3) Candidate vs Champion (Policy-Relevant)
Overall:
- Champion: PPE10 `0.3254`, MdAPE `0.1637`
- Candidate: PPE10 `0.2184`, MdAPE `0.2521`
- Delta: PPE10 `-0.1070`, MdAPE `+0.0885` (worse)

Major segments (candidate - champion):
- `ELEVATOR`: PPE10 `-0.0839`, MdAPE `+0.1632`
- `SINGLE_FAMILY`: PPE10 `-0.1463`, MdAPE `+0.0564`
- `SMALL_MULTI`: PPE10 `-0.0520`, MdAPE `+0.1146`
- `WALKUP`: PPE10 `-0.0626`, MdAPE `+0.0966`

Arena proposal (`reports/arena/proposal_267ceec7e870.json`):
- Status: `no_winner`
- Candidate gate_pass: `false`
- `weighted_segment_mdape_improvement=-0.6232` (policy requires `>= 0.05`) -> FAIL
- `overall_ppe10_lift=-0.1070` -> FAIL
- `max_major_segment_ppe10_drop=0.1463` (policy max `0.02`) -> FAIL
- `min_major_segment_ppe10=0.1447` (policy floor `0.24`) -> FAIL
- `drift_alert_delta=0` -> PASS
- `fairness_alert_delta=-1` -> PASS

## 4) Temporal Stability Tradeoffs
Computed on identical chronological quarter slices (`2023Q3` to `2024Q4`):
- MdAPE std: champion `0.0046` vs candidate `0.0107` (delta `+0.0061`, worse)
- PPE10 std: champion `0.0065` vs candidate `0.0070` (delta `+0.0005`, slightly worse)
- MdAPE range widened by `+0.0171` (worse)
- PPE10 range narrowed by `-0.0024` (slightly better), but absolute PPE10 levels were materially lower in every quarter.

Interpretation:
- Stability did not improve in a way that offsets the large accuracy regression.
- Candidate underperforms champion both overall and across major segments.

## 5) Release Validation
- Smoke: PASS
  - `reports/validation/v1_readiness_hseason001_smoke.json`
  - `reports/validation/v1_readiness_hseason001_smoke.md`
- Production: FAIL
  - `reports/validation/v1_readiness_hseason001_production.json`
  - `reports/validation/v1_readiness_hseason001_production.md`
  - Blocking gate: `arena_governance_production` (`latest proposal status=no_winner`)

## 6) Recommendation
Recommendation: **rollback** this branch from promotion candidacy.

Rationale:
- Fails core quality gates in `config/arena_policy.yaml`.
- No uplift on MdAPE/PPE10 overall or in major segments.
- Temporal stability is mixed-to-worse and does not justify deployment risk.

Suggested follow-up experiment:
- Keep time-aware split/diagnostics, but test a smaller temporal feature set (trend only or cyclical only), then re-run arena propose before considering combine branch work.

## 7) Full Command Log
1. `./scripts/ds_workflow.sh train-candidate --hypothesis-id H-SEASON-001 --change-type feature --change-summary "Added cyclical month, trend, and regime bucket features with time-slice diagnostics" --owner colby --feature-set-version v3.2_seasonality --dataset-version 20260223_sales_2019_2025_seasonv1 --input-csv data/processed/hseason001_train_20260223.csv --strategy global --artifact-tag hseason001_20260223 --run-name hseason001-temporal-features-20260223`
2. `./scripts/ds_workflow.sh arena-propose`
3. `python3 -m src.validate_release --mode smoke --mlflow-tracking-uri sqlite:///mlflow.db --arena-policy-path config/arena_policy.yaml --arena-dir reports/arena --output-md reports/validation/v1_readiness_hseason001_smoke.md --output-json reports/validation/v1_readiness_hseason001_smoke.json`
4. `python3 -m src.validate_release --mode production --mlflow-tracking-uri sqlite:///mlflow.db --arena-policy-path config/arena_policy.yaml --arena-dir reports/arena --output-md reports/validation/v1_readiness_hseason001_production.md --output-json reports/validation/v1_readiness_hseason001_production.json`
5. `python3 - <<PY ... compute champion vs candidate overall/segment/temporal deltas ... PY`

## 8) Artifact Paths
- Run card: `reports/arena/run_card_6612686186654a3685df7ffca7ca3bde.md`
- Candidate model: `models/model_v1_hseason001_20260223.joblib`
- Candidate metrics: `models/metrics_v1_hseason001_20260223.json`
- Candidate segment scorecard: `reports/model/segment_scorecard_v1_hseason001_20260223.csv`
- Candidate temporal scorecard: `reports/model/temporal_scorecard_v1_hseason001_20260223.csv`
- Candidate predictions: `reports/model/evaluation_predictions_v1_hseason001_20260223.csv`
- Candidate SHAP summary: `reports/model/shap_summary_v1_hseason001_20260223.png`
- Candidate SHAP waterfall: `reports/model/shap_waterfall_v1_hseason001_20260223.png`
- Arena proposal: `reports/arena/proposal_267ceec7e870.json`
- Arena comparison CSV: `reports/arena/comparison_20260223T144123Z.csv`
- Comparison summary JSON: `reports/arena/hseason001_compare_summary_20260223.json`
- Recomputed baseline temporal scorecard: `reports/model/temporal_scorecard_v1_baseline_recomputed.csv`
- Recomputed candidate temporal scorecard: `reports/model/temporal_scorecard_v1_hseason001_20260223_recomputed.csv`
- Smoke readiness report: `reports/validation/v1_readiness_hseason001_smoke.md`
- Smoke readiness JSON: `reports/validation/v1_readiness_hseason001_smoke.json`
- Production readiness report: `reports/validation/v1_readiness_hseason001_production.md`
- Production readiness JSON: `reports/validation/v1_readiness_hseason001_production.json`
