# H-SEG-TIER-001 Execution Report (Data + ML)

- Date: 2026-02-23
- Hypothesis: `H-SEG-TIER-001`
- Decision: **NO-GO** (do not promote)

## 1) Proxy Strategy (Inference-Safe)
`price_tier_proxy` is built only from inference-available non-target signals:
- `gross_square_feet`
- `building_age`
- `distance_to_center_km`
- `total_units` / `residential_units`
- `borough`

Implementation details:
- Compute non-target proxy score.
- Fit q25/q50/q75 bins per `property_segment`.
- Apply segment bins where available; otherwise global fallback.
- If proxy score inputs are missing, default to `core`.

Leakage controls:
- No use of `sale_price` or target-derived `price_tier` in proxy computation.
- Inference hard-blocks routing configs that use `price_tier`.

## 2) Implementation Summary
- Added non-leaky proxy module: `src/price_tier_proxy.py`
- ETL now materializes `price_tier_proxy` + source metadata.
- Model training now fits proxy bins on train split only and applies to test/inference.
- Inference derives `price_tier_proxy` when missing (using artifact bins).
- Database schema supports `price_tier_proxy` for `sales`.

## 3) Candidate Run
- Run ID: `d3bb60de10354f92be429f482fab2e4e`
- Model version: `3`
- Alias: `candidate`
- Strategy: `segmented_router`
- Router mode: `segment_plus_tier`
- Router columns: `property_segment`, `price_tier_proxy`
- Segment submodels: `16`
- Tags:
  - `dataset_version=20260223_sales_2019_2025_proxyv1`
  - `feature_set_version=v3.1_segment_plus_tier_proxy`

## 4) Performance Impact
Overall metrics:
- Champion: PPE10 `0.3254`, MdAPE `0.1637`
- H-SEG-001: PPE10 `0.2213`, MdAPE `0.2285`
- H-SEG-TIER-001: PPE10 `0.2249`, MdAPE `0.2292`

Arena deltas vs champion (`proposal_bd679c340e64.json`):
- weighted_segment_mdape_improvement: `-0.3858` (policy requires `>= 0.05`) -> FAIL
- overall_ppe10_lift: `-0.1005` -> FAIL
- max_major_segment_ppe10_drop: `0.1895` (policy max `0.02`) -> FAIL
- min_major_segment_ppe10: `0.1832` (policy floor `0.24`) -> FAIL
- drift_alert_delta: `0` -> PASS
- fairness_alert_delta: `-1` -> PASS

Incremental vs H-SEG-001:
- Weighted segment MdAPE relative change: `+3.42%` improvement vs H-SEG-001.
- PPE10 improves in 3/4 segments vs H-SEG-001, but still materially below champion.

## 5) Release Validation
- Smoke validation: PASS
  - `reports/validation/hsegtier001_smoke_readiness_20260223.json`
- Production validation: FAIL
  - `reports/validation/hsegtier001_production_readiness_20260223.json`
  - blocking check: `arena_governance_production` (`status=no_winner`)

## 6) Risk Summary
- Model quality remains below champion across core policy gates.
- Routing complexity increased (16 route models), but gains were not sufficient for promotion.
- ETL DB full-load path showed infra instability (Docker/Postgres connection reset); CSV-based training path remains reliable fallback.

## 7) Tests Executed
- `pytest -q tests/test_inference.py tests/test_etl.py tests/test_price_tier_proxy.py` -> 17 passed

Coverage intent:
- no target leakage (`test_proxy_is_not_target_derived`)
- inference-time proxy availability (`test_predict_dataframe_segment_plus_tier_derives_proxy_at_inference_time`)
- backward compatibility (`test_predict_dataframe_global`, `test_predict_dataframe_segmented_router_with_fallback`)
- target-derived routing blocked (`test_predict_dataframe_disallows_target_derived_price_tier_routing`)

## 8) Full Command Log
1. `sed -n '1,260p' docs/hypotheses/FEATURE_BACKLOG.md`
2. `sed -n '1,260p' docs/hypotheses/HYPOTHESIS_BACKLOG.md`
3. `sed -n '1,260p' docs/DATASET_FEATURE_CHANGE_PROCESS.md`
4. `rg -n "price_tier_proxy|price_tier|segment_plus_tier|router_mode|routing|model_route" src tests docs -S`
5. `python3 -m pytest -q tests/test_inference.py tests/test_etl.py tests/test_price_tier_proxy.py`
6. `python3 -m src.database create && python3 -m src.etl --input data/raw/annualized_sales_2019_2025.csv --replace-sales --write-report --report-tag hsegtier001_20260223` (failed; DB connection reset during insert)
7. `python3 - <<PY ... run_etl(dry_run=True) ... write data/processed/hsegtier001_train_20260223.csv ... PY`
8. `./scripts/ds_workflow.sh train-candidate --hypothesis-id H-SEG-TIER-001 --change-type architecture --change-summary "Segment + non-leaky price_tier_proxy router with global fallback" --owner colby --feature-set-version v3.1_segment_plus_tier_proxy --dataset-version 20260223_sales_2019_2025_proxyv1 --input-csv data/processed/hsegtier001_train_20260223.csv --strategy segmented_router --router-mode segment_plus_tier --min-segment-rows 1200 --artifact-tag hsegtier001_20260223 --run-name hsegtier001-segment-tier-router-20260223`
9. `./scripts/ds_workflow.sh arena-propose`
10. `./scripts/ds_workflow.sh arena-status`
11. `python3 -m src.validate_release --mode smoke --contract-profile canonical --output-md reports/validation/hsegtier001_smoke_readiness_20260223.md --output-json reports/validation/hsegtier001_smoke_readiness_20260223.json` (first attempt failed; Docker daemon unavailable)
12. `open -a Docker` (restart daemon)
13. `python3 -m src.validate_release --mode smoke --contract-profile canonical --output-md reports/validation/hsegtier001_smoke_readiness_20260223.md --output-json reports/validation/hsegtier001_smoke_readiness_20260223.json`
14. `python3 -m src.validate_release --mode production --contract-profile canonical --output-md reports/validation/hsegtier001_production_readiness_20260223.md --output-json reports/validation/hsegtier001_production_readiness_20260223.json`

## 9) Artifact Paths
- Strategy doc: `docs/hypotheses/H-FEAT-002-price-tier-proxy.md`
- Candidate run card: `reports/arena/run_card_d3bb60de10354f92be429f482fab2e4e.md`
- Candidate model: `models/model_v1_hsegtier001_20260223.joblib`
- Candidate metrics: `models/metrics_v1_hsegtier001_20260223.json`
- Candidate scorecard: `reports/model/segment_scorecard_v1_hsegtier001_20260223.csv`
- Candidate predictions: `reports/model/evaluation_predictions_v1_hsegtier001_20260223.csv`
- Candidate SHAP summary: `reports/model/shap_summary_v1_hsegtier001_20260223.png`
- Candidate SHAP waterfall: `reports/model/shap_waterfall_v1_hsegtier001_20260223.png`
- ETL dry-run evidence: `reports/data/etl_run_20260223_hsegtier001_dryrun_20260223.md`
- ETL dry-run stage CSV: `reports/data/etl_run_20260223_hsegtier001_dryrun_20260223.csv`
- Processed training dataset: `data/processed/hsegtier001_train_20260223.csv`
- Arena proposal JSON: `reports/arena/proposal_bd679c340e64.json`
- Arena proposal markdown: `reports/arena/proposal_bd679c340e64.md`
- Arena comparison CSV: `reports/arena/comparison_20260223T142544Z.csv`
- Smoke validation JSON: `reports/validation/hsegtier001_smoke_readiness_20260223.json`
- Smoke validation markdown: `reports/validation/hsegtier001_smoke_readiness_20260223.md`
- Production validation JSON: `reports/validation/hsegtier001_production_readiness_20260223.json`
- Production validation markdown: `reports/validation/hsegtier001_production_readiness_20260223.md`
