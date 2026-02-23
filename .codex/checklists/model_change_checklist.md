# Model Change Checklist
- [ ] Hypothesis documented
- [ ] Training features validated as inference-available (`validate_training_feature_contract`)
- [ ] No target-derived fields in model inputs or routing keys
- [ ] Leakage guard tests passing (`tests/test_inference.py`, `tests/test_model_feature_contracts.py`, `tests/test_price_tier_proxy.py`)
- [ ] Metrics and segment scorecard generated
- [ ] Missingness/drift diagnostics generated (`feature_missingness_*.csv`, `feature_drift_segment_time_*.csv`)
- [ ] Diagnostics reviewed for alerting slices and remediation plan captured
- [ ] SHAP artifacts generated
- [ ] Arena proposal/evidence updated
