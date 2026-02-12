# H-SEG-001: Segment-Only Router Challenger

Created: 2026-02-12  
Owner: colby  
Status: ready_to_run

---

## 1) Hypothesis ID
- `H-SEG-001`

## 2) Problem Statement
The current champion is a single global model. It underperforms in non-single-family segments, and overall accuracy is below operating thresholds.

Primary pain points:
- Overall PPE10 is low for production expectations.
- Segment performance is uneven.
- Monitoring and retrain policy both signal alert/retrain.

Business risk:
- Inconsistent valuation quality across property types can reduce trust and create biased operational behavior by segment.

## 3) Evidence (Current Champion)
- Date of evidence pull: 2026-02-12
- Champion alias: `spec-nyc-avm@champion` -> model version `1`
- Champion run ID: `34e917e198af4e58adb2097b8d9ca229`

Data quality context:
- ETL report: `reports/data/etl_run_20260211_prod.md`
- Feature-engineered rows: `295,457`
- Latest sale date in ETL run: `2024-12-31`

Champion metrics source:
- `models/metrics_v1.json`
- `reports/model/segment_scorecard_v1.csv`
- `reports/monitoring/performance_latest.json`

Baseline metrics:
- overall n: `59,092`
- overall PPE10: `0.3254`
- overall MdAPE: `0.1637`
- overall R2: `0.0281`

Per-segment (major):
- `ELEVATOR`: PPE10 `0.2285`, MdAPE `0.2224`, n `25,785`
- `SINGLE_FAMILY`: PPE10 `0.4585`, MdAPE `0.1088`, n `25,146`
- `SMALL_MULTI`: PPE10 `0.2147`, MdAPE `0.2381`, n `3,018`
- `WALKUP`: PPE10 `0.2248`, MdAPE `0.2313`, n `5,143`

Monitoring signals:
- `reports/monitoring/performance_latest.json` -> status `alert`
- `reports/monitoring/drift_latest.json` -> alerts `3`
- `reports/releases/retrain_decision_latest.json` -> `should_retrain: true`

## 4) EDA / Error Analysis Findings
What was inspected:
- Segment scorecard and per-segment errors.
- Monitoring alerts and retrain policy outputs.
- Arena run card risk notes.

What was observed:
- Global model appears to fit `SINGLE_FAMILY` materially better than other segments.
- Segment PPE10 spread is wide (`~0.244`), indicating heterogeneous behavior by property type.
- Drift/performance alerts indicate existing champion behavior is not stable enough for passive hold.

Interpretation:
- Property-type heterogeneity is likely strong enough that a single shared decision surface is too blunt.
- First architecture step should isolate segment routing before introducing tier routing complexity.

## 5) Hypothesis (Falsifiable)
If we replace the global-only prediction path with a segmented router using `property_segment`-specific submodels (with global fallback), then:

1. Weighted segment MdAPE should improve by at least `5%` relative to champion.
2. No major segment PPE10 should drop by more than `2%` absolute.
3. Major segment PPE10 floor should remain at or above `0.24`.
4. No additional drift/fairness alerts should be introduced.

These thresholds align with `config/arena_policy.yaml`.

## 6) Planned Change
- `change_type`: `architecture`
- strategy: `segmented_router`
- router_mode: `segment_only`
- fallback path: global model remains inside artifact for sparse/unknown routing keys

Training parameters:
- `min_segment_rows`: `2000`
- `feature_set_version`: `v3.0`
- `dataset_version`: `20260212`
- `optuna_trials`: `30` (or lower for smoke iteration)

Scope:
- This hypothesis only tests routing by segment.
- Tier routing is explicitly deferred to separate hypothesis (`H-SEG-TIER-001`) after this result.

## 7) Experiment Plan
### Pre-flight checks
```bash
./scripts/ds_workflow.sh daily
./scripts/ds_workflow.sh arena-status
```

Expected pre-flight:
- champion alias exists
- no pending proposal requiring resolution
- data/monitoring artifacts readable

### Candidate training command
```bash
./scripts/ds_workflow.sh train-candidate \
  --hypothesis-id H-SEG-001 \
  --change-type architecture \
  --change-summary "Segment-only router using property_segment with global fallback" \
  --owner colby \
  --feature-set-version v3.0 \
  --dataset-version 20260212 \
  --strategy segmented_router \
  --router-mode segment_only \
  --min-segment-rows 2000
```

### Post-train validation checks
```bash
./scripts/ds_workflow.sh arena-status
python3 - <<'PY'
import glob, json, pandas as pd
metrics = sorted(glob.glob("models/metrics_v1_*.json"))[-1]
preds = sorted(glob.glob("reports/model/evaluation_predictions_v1_*.csv"))[-1]
print("metrics:", metrics)
print("preds:", preds)
payload = json.load(open(metrics))
print("overall:", payload["overall"])
print("metadata:", {k: payload["metadata"].get(k) for k in ["model_strategy","router_mode","segment_model_count","min_segment_rows"]})
df = pd.read_csv(preds)
print(pd.crosstab(df["property_segment"], df["model_route"]))
PY
```

### Arena comparison and decision flow
```bash
./scripts/ds_workflow.sh arena-propose
./scripts/ds_workflow.sh arena-status
```

If proposal passes and review is acceptable:
```bash
./scripts/ds_workflow.sh arena-approve --proposal-id <proposal_id> --approved-by "colby"
```

Otherwise:
```bash
./scripts/ds_workflow.sh arena-reject --proposal-id <proposal_id> --reason "insufficient uplift or guardrail failure" --rejected-by "colby"
```

## 8) Risks / Rollback
Primary risks:
- Overfitting in smaller segments despite row floor.
- Apparent metric gains due to volatile slices.
- Operational complexity increase from routed architecture.

Known modeling caveat to track:
- `price_tier` remains a model input and is target-derived in ETL. This hypothesis does not change that; separate cleanup hypothesis is recommended.

Rollback:
- Reject proposal if gates fail.
- If already approved and regression observed, re-point champion alias to prior version.
- Keep prior champion run/version in change log for fast reversion.

## 9) Definition of Done for H-SEG-001
All must be true:
1. Candidate run tracked with full metadata and run card.
2. Arena proposal generated against champion.
3. Policy gates evaluated with no guardrail violations.
4. Human decision logged (approve/reject) with rationale.
5. If approved, production validation can still pass release checks.

## 10) Outcome (Fill After Run)
- Candidate run ID:
- Candidate model version:
- Proposal ID:
- Decision: approved / rejected
- Before vs after deltas:
  - weighted segment MdAPE improvement:
  - overall PPE10 delta:
  - max major-segment PPE10 drop:
  - drift alert delta:
  - fairness alert delta:
- Follow-up hypothesis:
