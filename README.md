# S.P.E.C. NYC

An applied data science and MLOps project for NYC residential AVM development.

I built this repository to show how I work as a production-minded data scientist: define hypotheses, build reliable data pipelines, train and evaluate models with segment-level accountability, and make controlled champion/challenger decisions with auditable evidence.

## Why I Built It

Most DS portfolios stop at notebooks and one model metric. This project is intentionally different.

I wanted to demonstrate a full lifecycle that reflects how teams operate in real environments:
- data ingestion and ETL with data contracts,
- reproducible model training and evaluation,
- explainability artifacts,
- model registry and promotion controls,
- monitoring, retrain policy, and release validation.

## What This Repository Demonstrates

1. Reliable data foundation:
- NYC Open Data connector with retries, pagination, caching, and annualized pulls (`src/connectors.py`).
- ETL with residential filtering, property ID normalization, duplicate handling, segmentation, imputation, and Postgres load (`src/etl.py`).
- explicit data-contract checks for schema, freshness, null thresholds, and domain constraints (`src/validation/data_contracts.py`).

2. Production-style modeling:
- baseline XGBoost training pipeline with time-based split and optional Optuna tuning (`src/model.py`).
- global and segmented routing strategies with fallback behavior.
- leakage controls around price-tier usage (routing safeguards and proxy requirement for tier-based routing).

3. Evaluation and explainability:
- overall and slice metrics (PPE10, MdAPE, R2) (`src/evaluate.py`).
- segment and tier scorecards.
- SHAP summary and waterfall artifacts (`src/explain.py`).

4. MLOps lifecycle controls:
- MLflow run tracking with structured change metadata (`src/mlops/track_run.py`).
- registry alias lifecycle (`champion`, `challenger`, `candidate`).
- arena-based promotion proposals with policy gates and human approval/rejection (`src/mlops/arena.py`, `config/arena_policy.yaml`).

5. Operational validation:
- drift and performance monitors (`src/monitoring/`).
- retrain policy outputs (`src/retrain_policy.py`).
- release readiness validation with evidence-aware checks (`src/validate_release.py`).

6. Stakeholder-facing product layer:
- Streamlit app that reads model/evaluation artifacts and supports valuation workflows (`app.py`).

## Current Status (As Of February 12, 2026)

The core v1 DS/MLOps workflow is implemented and operational for local/offline experimentation.

| Area | Status | Primary Evidence |
|---|---|---|
| Data connector + ETL | Implemented | `src/connectors.py`, `src/etl.py` |
| Data contracts + ETL tests | Implemented | `src/validation/data_contracts.py`, `tests/test_etl.py` |
| Baseline model + evaluation + SHAP | Implemented | `src/model.py`, `src/evaluate.py`, `src/explain.py` |
| Champion/challenger arena | Implemented | `src/mlops/track_run.py`, `src/mlops/arena.py` |
| Monitoring + retrain policy | Implemented | `src/monitoring/`, `src/retrain_policy.py` |
| Release validation | Implemented | `src/validate_release.py` |
| DS workflow runbooks | Implemented | `docs/DS_ROLE_WORKFLOW.md`, `docs/MLOPS_ARENA_WORKFLOW.md` |
| Next model improvements | Active backlog | `docs/hypotheses/HYPOTHESIS_BACKLOG.md`, `docs/hypotheses/FEATURE_BACKLOG.md` |

## How I Treat This As A Real DS Workflow

I treat model work as hypothesis-driven, not ad-hoc retraining.

For every meaningful experiment, I track:
- hypothesis ID,
- change type and rationale,
- dataset version,
- feature set version,
- owner,
- measurable pass/fail gates,
- promotion decision and rollback path.

Key process docs:
- DS operating model: `docs/DS_ROLE_WORKFLOW.md`
- Arena lifecycle: `docs/MLOPS_ARENA_WORKFLOW.md`
- Hypothesis template: `docs/hypotheses/HYPOTHESIS_TEMPLATE.md`
- Backlog and branching: `docs/hypotheses/HYPOTHESIS_BACKLOG.md`
- Dataset and feature change control: `docs/DATASET_FEATURE_CHANGE_PROCESS.md`

## Data And Feature Governance Principles

This project uses two required version labels for reproducibility:
- `dataset_version`: the exact data snapshot used for training.
- `feature_set_version`: the exact feature logic/version used by the model.

I separate exploratory feature work from promoted feature work:
- exploratory features can start in training code,
- durable features should be promoted into ETL once validated.

This keeps model lineage clear and makes artifact comparisons meaningful.

## Modeling Strategy

Current strategy support:
- global model,
- segmented router model (`segment_only`),
- staged support for `segment_plus_tier` with non-leaky tier proxy requirements.

Why XGBoost for baseline:
- strong performance on structured/tabular data,
- robust with mixed numeric/categorical pipelines,
- efficient iteration for hypothesis testing,
- SHAP compatibility for feature attribution.

## Artifact-First Project Layout

If you are reviewing this as a hiring manager or technical lead, these paths show the workflow quality quickly:

- data and ETL evidence: `reports/data/`
- model metrics and scorecards: `models/`, `reports/model/`
- arena decisions and change notes: `reports/arena/`
- monitoring outputs: `reports/monitoring/`
- release readiness reports: `reports/validation/`
- release policy artifacts: `reports/releases/`

## What I Plan To Improve Next

Current priority stream is model strengthening and feature expansion, including:
- safer tier proxy strategy for segmented routing,
- NYC-specific feature engineering (transit, liquidity, temporal regime signals),
- continued challenger evaluation through arena gates before promotion.

Tracked in:
- `docs/hypotheses/FEATURE_BACKLOG.md`
- `docs/hypotheses/HYPOTHESIS_BACKLOG.md`

## Project Positioning

This repo is the NYC production-data progression of my earlier S.P.E.C. valuation work. The emphasis here is less on demo UI and more on end-to-end DS execution quality, model governance, and reproducible MLOps behavior.

## Author

Colby Reichenbach

GitHub: [ColbyRReichenbach](https://github.com/ColbyRReichenbach)
LinkedIn: [colbyreichenbach](https://linkedin.com/in/colbyreichenbach)

## License

MIT License (`LICENSE`)
