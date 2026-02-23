# Demo Operator Runbook

## Purpose
Run a reliable stakeholder demo with auditable outputs and fallback behavior if services are unavailable.

## Standard Run Sequence
1. Refresh snapshot report:
   - `python3 -m src.demo_snapshot`
2. Run smoke validation:
   - `python3 -m src.validate_release --mode smoke --contract-profile canonical --output-md reports/validation/demo_smoke_readiness_latest.md --output-json reports/validation/demo_smoke_readiness_latest.json`
3. Launch app:
   - `python3 -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8504`

## Evidence You Should Have Open During Demo
- Demo summary:
  - `reports/demo/stakeholder_demo_snapshot_latest.md`
  - `reports/demo/stakeholder_demo_snapshot_latest.json`
- Governance:
  - latest `reports/arena/proposal_*.json`
  - latest `reports/arena/comparison_*.csv`
- Monitoring:
  - `reports/monitoring/drift_latest.json`
  - `reports/monitoring/performance_latest.json`
  - `reports/releases/retrain_decision_latest.json`
- Release gate status:
  - `reports/validation/demo_smoke_readiness_latest.json`

## Service-Unavailable Fallback Plan

### If Postgres is unavailable
- App already falls back to:
  - `data/raw/annualized_sales_2019_2025.csv`
- Keep the demo on artifact-backed views (metrics/governance/monitoring) and mention data source fallback in UI warning banner.

### If Streamlit cannot start
- Deliver demo from report artifacts only:
  - `reports/demo/stakeholder_demo_snapshot_latest.md`
  - `reports/arena/run_card_*.md`
  - `reports/arena/proposal_*.json`
  - `reports/monitoring/*.json`

### If Docker services are unavailable for smoke checks
- Run non-container evidence path:
  - `python3 -m src.demo_snapshot`
  - `python3 -m pytest -q`
- State explicitly that full smoke infra checks were skipped and provide last successful smoke artifact path.

## Operator Notes
- Do not claim promotion unless arena proposal status is `approved` and release Gate E is green.
- Keep references absolute to artifact paths shown in the app/report at demo time.
