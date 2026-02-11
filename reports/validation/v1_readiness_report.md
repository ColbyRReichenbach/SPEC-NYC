# S.P.E.C. NYC V1 Readiness Report

- Started (UTC): `2026-02-11T01:37:45.690476`
- Finished (UTC): `2026-02-11T01:39:32.682979`
- Duration: `106.99s`

## Checks

| Check | Status | Detail | Log |
|---|---|---|---|
| unit_tests | pass | exit=0; OK | reports/validation/logs/unit_tests.log |
| docker_compose_config | pass | exit=0; time="2026-02-10T20:37:57-05:00" level=warning msg="/Users/colbyreichenbach/Desktop/SPEC-NYC/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion" | reports/validation/logs/docker_compose_config.log |
| docker_compose_up_db | pass | exit=0;  Container spec_nyc_db  Running | reports/validation/logs/docker_compose_up_db.log |
| db_connectivity | pass | exit=0; db_ok | reports/validation/logs/db_connectivity.log |
| db_schema_create | pass | exit=0;   Base = declarative_base() | reports/validation/logs/db_schema_create.log |
| etl_smoke | pass | exit=0; 2026-02-10 20:38:05,684 - INFO - Wrote ETL CSV summary: reports/data/etl_run_20260211.csv | reports/validation/logs/etl_smoke.log |
| model_smoke | pass | exit=0; 2026-02-10 20:38:34,852 - INFO - Generated SHAP artifacts: {'summary_plot_path': 'reports/model/shap_summary_v1_smoke.png', 'waterfall_plot_path': 'reports/model/shap_waterfall_v1_smoke.png', 'sample_size': 120, 'explainer_type': 'xgboost_pred_contribs'} | reports/validation/logs/model_smoke.log |
| evaluate_smoke | pass | exit=0; Overall => n=160, PPE10=0.963, MdAPE=0.028, R2=0.985 | reports/validation/logs/evaluate_smoke.log |
| explain_smoke | pass | exit=0; 2026-02-10 20:38:50,935 - INFO - Saved SHAP waterfall plot: reports/model/shap_waterfall_v1_smoke.png | reports/validation/logs/explain_smoke.log |
| mlflow_track_smoke | pass | exit=0; 2026/02/10 20:38:56 INFO alembic.runtime.migration: Will assume non-transactional DDL. | reports/validation/logs/mlflow_track_smoke.log |
| drift_monitor_smoke | pass | exit=0; } | reports/validation/logs/drift_monitor_smoke.log |
| performance_monitor_smoke | pass | exit=0; } | reports/validation/logs/performance_monitor_smoke.log |
| retrain_policy_smoke | pass | exit=0; } | reports/validation/logs/retrain_policy_smoke.log |
| streamlit_app_smoke | pass | app process stayed healthy for startup window | reports/validation/logs/streamlit_app_smoke.log |
| artifact_inventory | pass | all required artifacts present | - |
| docker_compose_stop_db | pass | exit=0;  Container spec_nyc_db  Stopped | reports/validation/logs/docker_compose_stop_db.log |
| release_tag | pass | created git tag v1.0 | reports/validation/logs/release_tag_create.log |

## Gates

| Gate | Status | Failed Checks | Missing Checks |
|---|---|---|---|
| Gate A (Data) | done | - | - |
| Gate B (Model) | done | - | - |
| Gate C (Product) | done | - | - |
| Gate D (Ops) | done | - | - |
| Gate E (Release) | done | - | - |

## Artifacts

- JSON payload: `reports/validation/v1_readiness_report.json`
- Command logs: `reports/validation/logs`