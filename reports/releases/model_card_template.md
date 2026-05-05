# S.P.E.C. NYC AVM Model Card Template

Package ID:
Model version:
Dataset version:
Feature contract version:
Git SHA:

## Intended Use

- Candidate NYC borough-level AVM research model for governed valuation workflow demonstration.

## Prohibited Use

- Not an appraisal.
- Not approved for lending, tax assessment, insurance, or consumer-facing production decisions without formal model-risk review.

## Data Sources

- Source names and URIs:
- Raw row count:
- Post-filter row count:
- Data snapshot SHA-256:
- Known limitations:

## Training Window

- Minimum sale date:
- Maximum sale date:
- Training rows:

## Validation Window

- Holdout logic:
- Validation rows:

## Model Type

- Model class:
- Strategy:
- Router mode:
- Hyperparameters:

## Target

- Target column:
- Target transform:
- Optimization objective:

## Features

- Model features:
- Router columns:
- Feature contract version:

## Leakage Controls

- Target-derived fields blocked:
- Point-in-time controls:
- Inference availability controls:

## Performance

- Overall PPE10:
- Overall MdAPE:
- Overall R2:

## Slice Performance

- Borough metrics:
- Segment metrics:
- Price-band metrics:
- Major slice regressions:

## Confidence and Intervals

- Interval method:
- Interval coverage:
- Hit/no-hit policy:
- Calibration limitations:

## Fairness and Proxy Audit

- Proxy audit scope:
- Valuation-ratio gaps:
- Overvaluation/undervaluation gaps:
- Limitations:

## Limitations

- Public data limitations:
- Coverage limitations:
- Known missing signals:

## Known Failure Modes

- Sparse comps:
- Unusual property condition:
- Major renovation:
- Non-market sale:
- Out-of-distribution request:

## Monitoring Plan

- Feature drift:
- Prediction drift:
- Performance decay:
- Data freshness:
- Segment-level alerts:

## Rollback Plan

- Previous champion package:
- Rollback package:
- Rollback trigger:
- Owner:
