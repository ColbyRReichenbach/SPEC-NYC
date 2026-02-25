# Arena Lifecycle Release Memo (2026-02-23)

- Candidate run: `879ab7838c214d3a907e34a687978264` (model version `6`)
- Proposal: `57e6c66f5205` (`no_winner`)
- Recommendation: **REJECT**

## Policy Gates (Arena)
| Gate | Actual | Threshold | Status |
|---|---:|---:|---|
| weighted_segment_mdape_improvement | -0.7000811446896945 | >= 0.05 | fail |
| max_major_segment_ppe10_drop | 0.12676622572213903 | <= 0.02 | fail |
| major_segment_ppe10_floor | 0.12190202846837063 | >= 0.24 | fail |
| no_new_drift_alerts | 0 | <= 0 | pass |
| no_new_fairness_alerts | 0 | <= 0 | pass |

## Production Readiness
- Report JSON: `reports/validation/v1_readiness_arena_lifecycle_20260223T1602_production.json`
- Report MD: `reports/validation/v1_readiness_arena_lifecycle_20260223T1602_production.md`
- Gate E all green: `False`
- Failed checks: `arena_governance_production`

## Artifacts
- Run card: `reports/arena/run_card_879ab7838c214d3a907e34a687978264.md`
- Comparison CSV: `reports/arena/comparison_20260223T160234Z.csv`
- Proposal JSON: `reports/arena/proposal_57e6c66f5205.json`
- Proposal MD: `reports/arena/proposal_57e6c66f5205.md`
- Validation JSON: `reports/validation/v1_readiness_arena_lifecycle_20260223T1602_production.json`
- Validation MD: `reports/validation/v1_readiness_arena_lifecycle_20260223T1602_production.md`
- Unit tests log: `reports/validation/logs/unit_tests.log`
- Streamlit smoke log: `reports/validation/logs/streamlit_app_production.log`

## Exact Commands Run
1. `python3 -m src.mlops.track_run --metrics-json models/metrics_v1_hseason001_20260223_t1555.json --model-artifact models/model_v1_hseason001_20260223_t1555.joblib --scorecard-csv reports/model/segment_scorecard_v1_hseason001_20260223_t1555.csv --predictions-csv reports/model/evaluation_predictions_v1_hseason001_20260223_t1555.csv --experiment-name spec-nyc-avm --run-name release-owner-hseason001-20260223T1603Z --dataset-version ds_hseason001_train_20260223 --git-sha <git_sha> --tracking-uri sqlite:///mlflow.db --hypothesis-id H-SEASON-001 --change-type feature --change-summary "Release lifecycle registration for H-SEASON-001 temporal challenger" --owner colbyreichenbach --feature-set-version fs_seasonality_regime_v1 --register-model --registered-model-name spec-nyc-avm --alias candidate --run-kind train --arena-dir reports/arena` -> **PASS** (run_id=879ab7838c214d3a907e34a687978264, model_version=6)
2. `./scripts/ds_workflow.sh arena-propose` -> **PASS** (proposal_id=57e6c66f5205, status=no_winner)
3. `python3 -m src.validate_release --mode production --mlflow-tracking-uri sqlite:///mlflow.db --arena-policy-path config/arena_policy.yaml --arena-dir reports/arena --output-md reports/validation/v1_readiness_arena_lifecycle_20260223T1602_production.md --output-json reports/validation/v1_readiness_arena_lifecycle_20260223T1602_production.json` -> **FAIL** (gate_e_all_green=false; failed_check=arena_governance_production)
4. `./scripts/ds_workflow.sh arena-reject --proposal-id 57e6c66f5205 --reason "Reject: failed weighted MdAPE uplift, major-segment PPE10 drop/floor gates" --rejected-by colbyreichenbach` -> **FAIL** (workflow wrapper bug: passes unsupported args to arena reject)
5. `python3 -m src.mlops.arena reject --proposal-id 57e6c66f5205 --arena-dir reports/arena --reason "Reject: failed weighted MdAPE uplift, major-segment PPE10 drop/floor gates" --rejected-by colbyreichenbach` -> **FAIL** (proposal status=no_winner (not rejectable; only pending can reject))
