<div align="center">

# S.P.E.C. NYC

**Production-oriented AVM pipeline for NYC residential valuation**  
Hypothesis-driven Data Science + MLOps with contracts, governance, and release gates.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Modeling-FF6F00)](https://xgboost.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Postgres](https://img.shields.io/badge/Postgres-Data%20Store-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![pytest](https://img.shields.io/badge/pytest-Tested-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)](https://openai.com/)

Built by **Colby Reichenbach**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/colby-reichenbach/)
[![Portfolio](https://img.shields.io/badge/Portfolio%20-%20Check%20Out%20My%20Work?style=flat-square&label=Check%20Out%20My%20Work&color=4B9CD3)](https://colbyrreichenbach.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ColbyRReichenbach)

</div>

## Overview

S.P.E.C. NYC is an applied AVM program designed to mirror real production workflows rather than notebook-only experimentation.

The repository focuses on:
- contract-driven ingestion and ETL,
- reproducible model training/evaluation,
- explainability and segment accountability,
- champion/challenger governance,
- release readiness validation with auditable artifacts.

## What This Repository Demonstrates

1. Data foundation and contract controls
- NYC Open Data connector with retries, pagination, caching, and annualized pulls (`src/connectors.py`).
- ETL with residential filtering, property ID normalization, duplicate handling, segmentation, imputation, and Postgres load (`src/etl.py`).
- Canonical and data-contract checks for schema, freshness, null thresholds, and domain constraints (`src/validation/data_contracts.py`, `src/canonical/`).

2. Production-style modeling
- Baseline XGBoost training with time-aware split and optional Optuna tuning (`src/model.py`).
- Routing modes (`global`, `segment_only`, staged `segment_plus_tier` with non-leaky proxy requirements).
- Leakage controls and inference-availability checks in train-time workflows.

3. Evaluation and explainability
- Overall and slice metrics (PPE10, MdAPE, R2) (`src/evaluate.py`).
- Segment/tier scorecards and temporal diagnostics.
- SHAP summary and local waterfall artifacts (`src/explain.py`).

4. MLOps governance lifecycle
- MLflow run tracking with structured metadata (`src/mlops/track_run.py`).
- Model alias lifecycle (`champion`, `challenger`, `candidate`).
- Arena proposals with policy gates and approve/reject workflows (`src/mlops/arena.py`, `config/arena_policy.yaml`).

5. Operational validation
- Drift and performance monitoring (`src/monitoring/`).
- Retrain policy decisions (`src/retrain_policy.py`).
- Release validator for smoke/production-readiness checks (`src/validate_release.py`).

6. Product-facing layer
- Next.js dashboard with valuation, governance, monitoring, and copilot flows (`web/`).
- Canonical-contract boundary so frontend/BFF never depends on raw client schemas.

## Current Status (February 12, 2026)

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

## Reproducibility and Governance

Every meaningful experiment should track:
- `hypothesis_id`
- `change_type` and rationale
- `dataset_version`
- `feature_set_version`
- owner
- measurable pass/fail gates
- promotion decision + rollback path

Core process docs:
- `docs/DS_ROLE_WORKFLOW.md`
- `docs/MLOPS_ARENA_WORKFLOW.md`
- `docs/hypotheses/HYPOTHESIS_TEMPLATE.md`
- `docs/hypotheses/HYPOTHESIS_BACKLOG.md`
- `docs/DATASET_FEATURE_CHANGE_PROCESS.md`

## Artifact-First Layout

Review these paths first for signal:
- `reports/data/` data and ETL evidence
- `models/`, `reports/model/` metrics and scorecards
- `reports/arena/` proposal/comparison/run-card evidence
- `reports/monitoring/` drift/performance snapshots
- `reports/validation/` release readiness reports
- `reports/releases/` release-policy outputs
- `docs/demo/` stakeholder demo/operator guidance

## Codex Agentic Extension

For autonomous, low-touch execution workflows:
- `docs/CODEX_AUTONOMY_SETUP.md`
- `docs/AVM_BUSINESS_LOGIC.md`
- `.codex/workflows/`

Implemented extension surface includes:
- codex role/workflow/checklist control plane (`.codex/`),
- datasource adapters and mappings (`src/datasources/`),
- canonical schema + portable contracts (`src/canonical/`),
- ETL/validator contract-profile flags,
- autopilot loop orchestration (`scripts/autonomy_loop.sh`).

## Project Positioning

This repository is the NYC production-data progression of earlier S.P.E.C. valuation work, with emphasis on DS execution quality, governance rigor, and reproducible MLOps behavior.

## License

MIT License (`LICENSE`)
