# Retrain Decision

- Decision: `retrain`
- Should Retrain: `True`
- Reasons: `performance monitor is alert; overall PPE10 below threshold (0.3254 < 0.7500); overall MdAPE above threshold (0.1637 > 0.0800); drift alerts exceed policy (3 > 0)`
- Signals: `{'drift_alerts': 3, 'performance_status': 'alert', 'model_age_days': 0}`