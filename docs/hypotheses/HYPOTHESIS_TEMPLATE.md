# Hypothesis Template

Use this before running `train-candidate`.

File naming:
- `docs/hypotheses/H-XXX-short-name.md`
- Example: `docs/hypotheses/H-SEG-001-segment-router.md`

---

## 1) Hypothesis ID
- `H-XXX`

## 2) Problem Statement
- What is underperforming?
- In which slice (segment, borough, price band, date window)?

## 3) Evidence (Current Champion)
- Date:
- Champion version / run ID:
- Baseline metrics:
  - overall PPE10:
  - overall MdAPE:
  - slice PPE10:
  - slice MdAPE:

## 4) EDA / Error Analysis Findings
- What did you inspect?
  - residuals by segment / tier / borough
  - feature missingness / shifts
  - recent data drift
- What did you observe?

## 5) Hypothesis (Falsifiable)
- If we change: `<specific change>`
- Then metric `<name>` in slice `<slice>` should improve by `<target>`
- While not degrading `<guardrail metric>` by more than `<threshold>`

## 6) Planned Change
- `change_type`: `feature | objective | data | architecture | tuning`
- Strategy:
  - `global`
  - `segmented_router` + `segment_only`
  - `segmented_router` + `segment_plus_tier`
- Parameters:
  - `min_segment_rows`:
  - `feature_set_version`:
  - `dataset_version`:

## 7) Experiment Plan
- Command to run:
```bash
./scripts/ds_workflow.sh train-candidate ...
```
- Arena evaluation plan:
```bash
./scripts/ds_workflow.sh arena-propose
./scripts/ds_workflow.sh arena-status
```

## 8) Risks / Rollback
- Risks:
- Rollback action:
  - Reject proposal in arena OR re-point champion alias to prior version.

## 9) Outcome (Fill After Run)
- Candidate run ID:
- Proposal ID:
- Decision: approved / rejected
- Before vs after:
  - overall PPE10 delta:
  - overall MdAPE delta:
  - key slice deltas:
- Notes / next hypothesis:
