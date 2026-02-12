# Dataset and Feature Change Process

Purpose:
- Keep training reproducible when data/features change.
- Make model lineage precise (`what data`, `which features`, `which code`).

---

## 1) Two Version Labels (Use Both)

1. `dataset_version`
- Identifies the data snapshot used to train.
- Example: `20260212_sales_2019_2025`

2. `feature_set_version`
- Identifies the transformation/feature definition set.
- Example: `v3.0_segment_router`

Rule:
- Data changes -> bump `dataset_version`.
- Feature logic changes -> bump `feature_set_version`.

---

## 2) Where to Compute New Features

Choose one path explicitly:

1. Compute in ETL (materialized feature in dataset)
- Use when feature is reused across models, monitoring, and explainability.
- Preferred for expensive or shared transformations.

2. Compute in training pipeline (ephemeral feature)
- Use for fast experimentation.
- Promote to ETL only after feature proves value.

Guideline:
- If feature is likely long-lived and operationally important, put it in ETL.
- If feature is exploratory, keep in model pipeline first.

---

## 3) Required Artifacts Per Change

For any feature/data change:
1. Hypothesis doc (`docs/hypotheses/H-*.md`)
2. ETL report (if ETL touched): `reports/data/etl_run_*.md`
3. Candidate run card: `reports/arena/run_card_*.md`
4. Model metrics + scorecard artifacts
5. Arena proposal/decision artifacts

---

## 4) Dataset Precision Checklist (Before Training)

1. Confirm data source:
- Postgres `sales` table or explicit `--input-csv`.

2. Confirm snapshot range:
- record min/max `sale_date`, row count.

3. Confirm feature schema:
- list feature columns used by model.

4. Confirm version labels:
- `dataset_version` set in command.
- `feature_set_version` set in command.

5. Confirm no leakage:
- no target-derived fields in model inputs or routing keys.

---

## 5) Local Workflow Notes (Docker vs No Docker)

1. Docker DB needed only if training from Postgres (`sales` table).
2. Docker DB not required if you pass `--input-csv` with all required training columns.

Recommended local default:
- Use Postgres + ETL for production-like runs.
- Use CSV for rapid isolated experiments.

---

## 6) Example Command (Production-Like Candidate)

```bash
./scripts/ds_workflow.sh train-candidate \
  --hypothesis-id H-SEG-001 \
  --change-type architecture \
  --change-summary "Segment-only router" \
  --owner colby \
  --feature-set-version v3.0 \
  --dataset-version 20260212_sales_2019_2025 \
  --strategy segmented_router \
  --router-mode segment_only \
  --min-segment-rows 2000
```

---

## 7) Governance Rule

No feature/data change is considered complete until:
1. version labels are set,
2. artifacts are produced,
3. arena decision is logged.

This is the minimum bar for production-grade traceability.
