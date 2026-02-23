# 10-Minute Stakeholder Demo Script (Ingestion -> Model -> Governance -> Monitoring)

## Objective
Show an auditable end-to-end narrative in one flow:
1. Ingestion quality and valuation context
2. Model output + evidence lineage
3. Governance/promotion decision state
4. Monitoring/retrain posture

## Pre-Demo Checklist (2-3 min)
- [ ] Generate latest snapshot report:
  - `python3 -m src.demo_snapshot`
- [ ] Start app:
  - `python3 -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8504`
- [ ] Confirm core evidence artifacts exist:
  - `reports/demo/stakeholder_demo_snapshot_latest.md`
  - `models/metrics_v1.json`
  - `reports/arena/proposal_*.json` (latest)
  - `reports/monitoring/drift_latest.json`
  - `reports/monitoring/performance_latest.json`
  - `reports/releases/retrain_decision_latest.json`

## Walkthrough Script (10 min)

### Minute 0-2: Ingestion and Product Context
- In app, show `Data Recency and Pipeline Status`.
- Narration:
  - "This view confirms data and ETL evidence are present before valuation claims are made."
  - "We keep artifact-first traceability: ETL, model, and monitoring outputs are file-backed."
- Evidence:
  - `reports/data/etl_run_*.md`
  - `reports/data/etl_run_*.csv`

### Minute 2-5: Valuation Output and Model Evidence
- In app, show `Per-Property Valuation and Explainability` and `Valuation Output Contract`.
- Narration:
  - "Product output includes `predicted_value`, bounds, confidence proxy, and top drivers."
  - "Values are tied to model/version metadata and explainability artifacts."
- Evidence:
  - `models/metrics_*.json`
  - `models/model_*.joblib`
  - `reports/model/segment_scorecard_*.csv`
  - `reports/model/shap_summary_*.png`
  - `reports/model/shap_waterfall_*.png`

### Minute 5-7: Governance and Promotion Decision
- In app, show `Governance and Monitoring Snapshot` -> `Promotion Decision State`.
- Narration:
  - "Arena proposal state drives promotion decisions; no promotion claim outside policy gates."
  - "Champion/candidate context and decision state are shown with proposal evidence."
- Evidence:
  - `reports/arena/proposal_*.json`
  - `reports/arena/comparison_*.csv`
  - `reports/arena/run_card_*.md`

### Minute 7-9: Monitoring and Retrain Policy
- In app, show `Governance and Monitoring Snapshot` -> `Drift and Performance`.
- Narration:
  - "Drift/performance are monitored independently from model training."
  - "Retrain decision captures policy-based operator action."
- Evidence:
  - `reports/monitoring/drift_latest.json`
  - `reports/monitoring/performance_latest.json`
  - `reports/releases/retrain_decision_latest.json`

### Minute 9-10: Close with Report Artifact
- Open and show:
  - `reports/demo/stakeholder_demo_snapshot_latest.md`
- Narration:
  - "This report is the meeting handoff artifact; every claim references generated evidence paths."

## Demo Exit Criteria
- All four narrative blocks were shown in app or report.
- Every claim referenced a concrete artifact path.
- No claim of promotion unless latest arena proposal status is approved and release gates are green.
