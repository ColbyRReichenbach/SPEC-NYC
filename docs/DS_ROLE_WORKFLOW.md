# S.P.E.C. NYC Real-World DS Workflow

This is the operating model to use if you want this repo to feel like a real data scientist role in production.

## Tooling Pattern (What Real Teams Use)

You work across three surfaces, not one:

1. Development surface:
- IDE + notebooks for feature experiments, error analysis, and code changes.
- Git branches + pull requests for controlled changes.

2. Experiment and model surface:
- MLflow for runs, metrics, artifacts, and model versions.
- Model Registry aliases for lifecycle state (`champion`, `challenger`, `candidate`).

3. Operations surface:
- Monitoring outputs (drift/performance/retrain policy).
- Release validation report + gate checks.
- Product-facing app for stakeholder review.

## Your Human-in-the-Loop Responsibilities

Automation runs the mechanics. You own the decisions:

1. Define hypothesis:
- What is failing now?
- What changed (features/objective/data/tuning)?
- What metric should improve and by how much?

2. Review arena proposal:
- Does challenger improve target metrics?
- Are there segment regressions or fairness/drift alerts?
- Approve or reject promotion.

3. Decide release:
- Confirm production gates are green.
- Confirm change narrative exists and is clear.

## Operating Cadence

You do not train/promote every day. Typical rhythm:

1. Daily (15-45 min):
- Run health checks.
- Review monitors, incidents, and arena status.

2. 2-3 times per week:
- Run new candidate experiments tied to explicit hypotheses.
- Compare challengers and create proposals.

3. Weekly:
- Champion/challenger review.
- Promotion decision if gates pass.
- Production validation before release.

## Repo Commands (Standard Workflow)

Before any candidate run, create a hypothesis doc:

- Template: `docs/hypotheses/HYPOTHESIS_TEMPLATE.md`
- Example file: `docs/hypotheses/H-SEG-001-segment-router.md`
- Backlog/lineage tracker: `docs/hypotheses/HYPOTHESIS_BACKLOG.md`
- Feature backlog: `docs/hypotheses/FEATURE_BACKLOG.md`
- Dataset/feature change control: `docs/DATASET_FEATURE_CHANGE_PROCESS.md`

Single entrypoint:

```bash
./scripts/ds_workflow.sh --help
```

Daily checks:

```bash
./scripts/ds_workflow.sh daily
```

One-time bootstrap (if no champion alias exists yet):

```bash
./scripts/ds_workflow.sh bootstrap-champion
```

Train and register a candidate (human-authored metadata required):

```bash
./scripts/ds_workflow.sh train-candidate \
  --hypothesis-id H-101 \
  --change-type feature \
  --change-summary "Added segmented sqft interactions by property type" \
  --owner colby \
  --feature-set-version v2.0 \
  --dataset-version 20260211 \
  --strategy segmented_router \
  --router-mode segment_only \
  --min-segment-rows 2000
```

For a follow-up challenger on segment+tier routing:

```bash
./scripts/ds_workflow.sh train-candidate \
  --hypothesis-id H-102 \
  --change-type architecture \
  --change-summary "Segment + price-tier router" \
  --owner colby \
  --feature-set-version v2.1 \
  --dataset-version 20260211 \
  --strategy segmented_router \
  --router-mode segment_plus_tier \
  --min-segment-rows 1200
```

Note:
- `segment_plus_tier` requires non-leaky `price_tier_proxy` in training data.
- Do not route on target-derived `price_tier`.

Propose and inspect promotion:

```bash
./scripts/ds_workflow.sh arena-propose
./scripts/ds_workflow.sh arena-status
```

Approve or reject:

```bash
./scripts/ds_workflow.sh arena-approve --proposal-id <proposal_id> --approved-by "colby"
./scripts/ds_workflow.sh arena-reject --proposal-id <proposal_id> --reason "segment regression" --rejected-by "colby"
```

Run production gate validation:

```bash
./scripts/ds_workflow.sh release-check
```

Open MLflow UI for run/model comparison:

```bash
./scripts/ds_workflow.sh mlflow-ui --port 5001
```

## What This Gives You for DS Job Readiness

1. Hypothesis-driven experimentation instead of random retraining.
2. Traceable model lineage and change narratives.
3. Controlled promotions with measurable rollback path.
4. A realistic split between technical execution and human decision ownership.
