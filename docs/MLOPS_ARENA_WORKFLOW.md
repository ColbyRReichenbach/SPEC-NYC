# MLOps Arena Workflow

This document describes the implemented champion/challenger lifecycle for S.P.E.C. NYC.

## Components

- Tracking and registration: `src/mlops/track_run.py`
- Arena lifecycle: `src/mlops/arena.py`
- Policy: `config/arena_policy.yaml`
- Arena artifacts: `reports/arena/`

Operator shortcut:
- `scripts/ds_workflow.sh` wraps the end-to-end DS lifecycle commands.

## 1) Track and Register a Candidate Run

Use `track_run` after training to log metrics/artifacts and optionally register a model version.

```bash
python3 -m src.mlops.track_run \
  --metrics-json models/metrics_v1.json \
  --model-artifact models/model_v1.joblib \
  --scorecard-csv reports/model/segment_scorecard_v1.csv \
  --predictions-csv reports/model/evaluation_predictions_v1.csv \
  --run-name train-v2-featurewave \
  --dataset-version 20260211 \
  --hypothesis-id H-021 \
  --change-type feature \
  --change-summary "Added spatiotemporal lag features" \
  --owner ml-engineer \
  --feature-set-version v2.1 \
  --register-model \
  --registered-model-name spec-nyc-avm \
  --alias candidate \
  --run-kind train
```

Outputs include a run card:
- `reports/arena/run_card_<run_id>.md`

## 2) Bootstrap the First Champion (one-time)

If no champion alias exists yet, run `track_run` once with:

```bash
--register-model --alias champion
```

This seeds the first production baseline.

## 3) Propose a Promotion

Arena compares recent challengers against champion and writes a promotion proposal.

```bash
python3 -m src.mlops.arena propose \
  --tracking-uri sqlite:///mlflow.db \
  --policy-path config/arena_policy.yaml \
  --experiment-name spec-nyc-avm
```

Artifacts:
- `reports/arena/comparison_<timestamp>.csv`
- `reports/arena/proposal_<proposal_id>.json`
- `reports/arena/proposal_<proposal_id>.md`

## 4) Review Current State

```bash
python3 -m src.mlops.arena status \
  --tracking-uri sqlite:///mlflow.db \
  --policy-path config/arena_policy.yaml
```

Shows:
- `champion/challenger/candidate` aliases
- latest proposal

## 5) Approve or Reject

Approve (promotes champion alias):

```bash
python3 -m src.mlops.arena approve \
  --proposal-id <proposal_id> \
  --tracking-uri sqlite:///mlflow.db \
  --approved-by "<name>"
```

Reject (keeps champion unchanged):

```bash
python3 -m src.mlops.arena reject \
  --proposal-id <proposal_id> \
  --reason "insufficient uplift in major segments" \
  --rejected-by "<name>"
```

Approval artifacts:
- `reports/arena/promotion_note_<proposal_id>.md`
- `reports/arena/model_change_log.md`

## 6) Production Release Gating

`src/validate_release.py --mode production` now checks:

1. Champion alias exists.
2. Latest arena proposal is approved and not expired.
3. Champion meets policy quality thresholds.
4. `reports/arena/model_change_log.md` exists.
