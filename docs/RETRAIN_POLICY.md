# S.P.E.C. NYC Retrain Policy

## Purpose
Define objective triggers for retraining the production AVM model.

## Signals
- Performance degradation:
  - Trigger if `PPE10 < 0.75`
  - Trigger if `MdAPE > 0.08`
- Drift degradation:
  - Trigger if any feature has drift status `alert`
  - Drift alert defaults: `PSI >= 0.25` or `KS >= 0.20`
- Model staleness:
  - Trigger if model age exceeds `90 days`

## Decision
- If any trigger is active -> `retrain`
- Otherwise -> `hold`

## Automation
- Run `src/monitoring/drift.py` and `src/monitoring/performance.py`
- Evaluate decision with `src/retrain_policy.py`
- Store decision output in `reports/releases/retrain_decision_latest.json`

## Overrides
- Human override must include:
  - reason,
  - approver,
  - expiry date for override.
