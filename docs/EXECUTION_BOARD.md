# S.P.E.C. NYC Execution Board

This board is the canonical execution tracker for V1.0 delivery.

## Rules

- Update status only after validation evidence is captured.
- Link every completed task to a concrete artifact or command output.
- Do not advance to the next gate until current gate criteria are all green.

## Status Legend

- `todo`
- `in_progress`
- `blocked`
- `done`

## Backlog

| Task ID | Workstream | Owner | ETA | Dependencies | Validation Command(s) | Evidence Link | Status |
|---|---|---|---|---|---|---|---|
| TASK-001 | W0 Governance | project-lead | 2026-02-11 | None | `rg -n "^### 1\\.[0-9]+" docs/NYC_IMPLEMENTATION_PLAN.md` | `docs/NYC_IMPLEMENTATION_PLAN.md` | done |
| TASK-002 | W1 Data Reliability | data-engineer | 2026-02-11 | TASK-001 | `python3 -m unittest tests.test_etl -v` | `src/validation/data_contracts.py`, `tests/test_etl.py`, `reports/data/etl_run_20260211.md` | done |
| TASK-003 | W2 Modeling Baseline | ml-engineer | 2026-02-11 | TASK-002 | `python3 -m src.model --input-csv /tmp/spec_model_smoke.csv --model-version v1 --optuna-trials 1 --shap-sample-size 200` | `src/model.py`, `models/model_v1.joblib`, `models/metrics_v1.json` | done |
| TASK-004 | W2 Modeling Baseline | ml-engineer | 2026-02-11 | TASK-003 | `python3 -m src.evaluate --predictions-csv reports/model/evaluation_predictions_v1.csv --output-json models/metrics_v1.json --segment-scorecard-csv reports/model/segment_scorecard_v1.csv` | `src/evaluate.py`, `reports/model/segment_scorecard_v1.csv` | done |
| TASK-005 | W2 Explainability | ml-engineer | 2026-02-11 | TASK-003 | `python3 -m src.explain --model-path models/model_v1.joblib --evaluation-csv reports/model/evaluation_predictions_v1.csv --summary-plot-path reports/model/shap_summary_v1.png --waterfall-plot-path reports/model/shap_waterfall_v1.png --sample-size 200` | `src/explain.py`, `reports/model/shap_summary_v1.png`, `reports/model/shap_waterfall_v1.png` | done |
| TASK-006 | W3 Dashboard | project-lead, ml-engineer | 2026-02-11 | TASK-004, TASK-005 | `python3 -m streamlit run app.py --server.headless true --server.port 8502` | `app.py`, `docs/NYC_IMPLEMENTATION_PLAN.md` | done |
| TASK-007 | W4 AI Security | ai-security | 2026-02-11 | TASK-006 | `python3 -m unittest tests.test_ai_security -v` | `src/ai_security.py`, `tests/test_ai_security.py`, `config/settings.py` | done |
| TASK-008 | W5 MLOps | ml-engineer | 2026-02-11 | TASK-004 | `python3 -m src.mlops.track_run --metrics-json models/metrics_v1.json --model-artifact models/model_v1.joblib --scorecard-csv reports/model/segment_scorecard_v1.csv --predictions-csv reports/model/evaluation_predictions_v1.csv --run-name w5-manual-track` | `src/mlops/track_run.py`, `mlflow.db`, `reports/releases/model_card_template.md`, `reports/releases/release_template.md` | done |
| TASK-009 | W5 Monitoring | data-engineer, ml-engineer | 2026-02-11 | TASK-008 | `python3 -m src.monitoring.drift --reference-csv reports/monitoring/reference_slice_v1.csv --current-csv reports/monitoring/current_slice_v1.csv && python3 -m src.monitoring.performance --predictions-csv reports/model/evaluation_predictions_v1.csv && python3 -m src.retrain_policy` | `src/monitoring/drift.py`, `src/monitoring/performance.py`, `src/retrain_policy.py`, `reports/monitoring/`, `reports/releases/retrain_decision_latest.json` | done |
| TASK-010 | W6 Validation & Release | validator | 2026-02-11 | TASK-001..TASK-009 | `python3 -m src.validate_release --tag-release` | `reports/validation/v1_readiness_report.md`, `reports/validation/v1_readiness_report.json`, `reports/validation/logs/release_tag_create.log`, `git tag v1.0` | done |

## Gate Checklist

| Gate | Criteria | Evidence | Status |
|---|---|---|---|
| Gate A (Data) | Contracts + ETL tests + ETL run report | `tests/test_etl.py`, `reports/data/etl_run_20260211.md` | done |
| Gate B (Model) | Baseline model + segment scorecard | `models/model_v1.joblib`, `models/metrics_v1.json`, `reports/model/segment_scorecard_v1.csv` | done |
| Gate C (Product) | Functional dashboard with live outputs | `app.py` runtime | done |
| Gate D (Ops) | MLflow + monitors + runbook | `mlflow.db`, `src/monitoring/`, `reports/monitoring/`, `reports/releases/` | done |
| Gate E (Release) | Validator all-green readiness report + release tag | `reports/validation/v1_readiness_report.md`, `git tag v1.0` | done |
