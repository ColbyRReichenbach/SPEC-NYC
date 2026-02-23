# S.P.E.C. NYC V1 Readiness Report

- Mode: `smoke`
- Started (UTC): `2026-02-23T15:51:23.810293`
- Finished (UTC): `2026-02-23T15:52:09.462659`
- Duration: `45.65s`

## Checks

| Check | Evidence Type | Status | Detail | Log |
|---|---|---|---|---|
| unit_tests | smoke | pass | exit=0; OK | reports/validation/logs/unit_tests.log |
| docker_compose_config | smoke | pass | exit=0; time="2026-02-23T10:51:27-05:00" level=warning msg="The \"OPENAI_API_KEY\" variable is not set. Defaulting to a blank string." | reports/validation/logs/docker_compose_config.log |
| docker_compose_up_db | smoke | pass | exit=0;  Container spec_nyc_db  Started | reports/validation/logs/docker_compose_up_db.log |
| db_connectivity | smoke | pass | exit=0; db_ok | reports/validation/logs/db_connectivity.log |
| db_schema_create | smoke | pass | exit=0;   Base = declarative_base() | reports/validation/logs/db_schema_create.log |
| canonicalization_smoke | smoke | pass | canonical contracts passed | - |
| etl_smoke | smoke | pass | exit=0; 2026-02-23 10:51:30,772 - INFO - Wrote ETL CSV summary: reports/data/etl_run_20260223_w6_smoke.csv | reports/validation/logs/etl_smoke.log |
| model_smoke | smoke | pass | exit=0; 2026-02-23 10:51:39,755 - INFO - Generated SHAP artifacts: {'summary_plot_path': 'reports/model/shap_summary_v1_smoke.png', 'waterfall_plot_path': 'reports/model/shap_waterfall_v1_smoke.png', 'sample_size': 120, 'explainer_type': 'xgboost_pred_contribs', 'explain_scope': 'global', 'model_strategy': 'global'} | reports/validation/logs/model_smoke.log |
| evaluate_smoke | smoke | pass | exit=0; Overall => n=160, PPE10=0.631, MdAPE=0.068, R2=0.872 | reports/validation/logs/evaluate_smoke.log |
| explain_smoke | smoke | pass | exit=0; 2026-02-23 10:51:47,956 - INFO - Saved SHAP waterfall plot: reports/model/shap_waterfall_v1_smoke.png | reports/validation/logs/explain_smoke.log |
| mlflow_track_smoke | smoke | pass | exit=0; 2026/02/23 10:51:51 INFO alembic.runtime.migration: Will assume non-transactional DDL. | reports/validation/logs/mlflow_track_smoke.log |
| drift_monitor_smoke | smoke | pass | exit=0; } | reports/validation/logs/drift_monitor_smoke.log |
| performance_monitor_smoke | smoke | pass | exit=0; } | reports/validation/logs/performance_monitor_smoke.log |
| retrain_policy_smoke | smoke | pass | exit=0; } | reports/validation/logs/retrain_policy_smoke.log |
| streamlit_app_smoke | smoke | pass | app process stayed healthy for startup window | reports/validation/logs/streamlit_app_smoke.log |
| artifact_inventory | smoke | pass | all required artifacts present | - |
| docker_compose_stop_db | smoke | pass | exit=0;  Container spec_nyc_db  Stopped | reports/validation/logs/docker_compose_stop_db.log |
| release_tag | release | pass | tagging skipped (--tag-release not set) | - |

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