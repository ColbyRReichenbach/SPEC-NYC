# AVM Business Logic & Real Estate Domain Playbook

## Objective
Embed durable AVM domain intelligence into the repo so agents can make better modeling and product decisions with minimal human intervention.

---

## 1) AVM Problem Definition

An AVM estimates likely market value for a property at a specific point in time.

### Practical output types
- Point estimate (expected value)
- Interval estimate (confidence band)
- Ranking utility (relative pricing confidence)

### Recommended product output for Azuli context
- `predicted_value`
- `low_value_bound` / `high_value_bound`
- `confidence_score`
- `top_drivers` (human-readable explanation)

---

## 2) Core Real Estate Signal Families

1. **Property fundamentals**
   - beds, baths, interior sqft, lot sqft, property type, year built, condition.
2. **Transaction dynamics**
   - last sale date/price, turnover frequency, recent neighborhood sales.
3. **Listing dynamics**
   - days on market, list-to-sale spread, price cut patterns.
4. **Location and accessibility**
   - submarket, school quality proxies, transit access, zoning context.
5. **Temporal regime features**
   - interest-rate environment, seasonality, macro trend shifts.

---

## 3) Data Quality Rules for AVM Reliability

Minimum controls before model training:
- deduplicate properties and transactions,
- enforce timestamp consistency,
- filter impossible/invalid values,
- prevent post-outcome leakage,
- require segment-level sample minimums.

### Leakage examples to block
- appraisal fields produced after contract close,
- closing-specific concessions unavailable at inference time,
- direct target transforms leaked from valuation outputs.

---

## 4) Modeling Strategy Guidance

### Baseline stack
- Gradient boosting model for tabular baseline.
- Segment-aware routing where sufficient data exists.
- Fallback global model for sparse segments.

### Production guidance
- train with time-based split (not random-only),
- evaluate by geography/price tier/property class,
- track interval calibration (not just point error),
- monitor drift and performance decay with alerts.

---

## 5) AVM Metrics That Matter to Stakeholders

Use both DS and business metrics:
- MdAPE / MAPE-like error metrics,
- PPE thresholds (e.g., within ±10%),
- segment stability over time,
- overvaluation vs undervaluation asymmetry,
- coverage of high-confidence predictions.

---

## 6) Governance and Risk Controls

1. **Policy gate before promotion**
   - candidate must beat champion on aggregate and critical slices.
2. **Release readiness gate**
   - no unresolved data contract violations,
   - explainability artifacts generated,
   - rollback path documented.
3. **Monitoring gate**
   - drift thresholds and retrain triggers predefined.

---

## 7) Adaptation to Unknown Internship Data

Given unknown schema and source quality, agents should:

1. Build canonical mapping first.
2. Classify each field by inference-time availability.
3. Compute missingness and stability profiles.
4. Generate automated feature candidacy report.
5. Train only after passing contract and leakage checks.

This avoids brittle, dataset-specific assumptions.

---

## 8) Frontend Requirements for Demonstration

To showcase production readiness:
- branded valuation UI,
- clear confidence and driver explanations,
- guardrails for out-of-scope inputs,
- downloadable prediction report (JSON/PDF).

Recommended views:
1. Single property valuation.
2. Batch valuation upload.
3. Monitoring snapshot (drift + recent performance).
4. Model/version metadata panel.

---

## 9) Agent Knowledge Injection Pattern

Place this file in prompt context for data/model/release agents.

Also maintain:
- `docs/hypotheses/HYPOTHESIS_BACKLOG.md` for experiment queue,
- `docs/DATASET_FEATURE_CHANGE_PROCESS.md` for controlled changes,
- `docs/RETRAIN_POLICY.md` for lifecycle behavior.

This keeps domain knowledge coupled with execution policy.

