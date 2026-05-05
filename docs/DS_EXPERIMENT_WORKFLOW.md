# Governed DS Experiment Workflow

Date: 2026-05-05

Purpose: define how a data scientist should work in this AVM platform so research intent, EDA, training, review, and model evidence are reproducible and auditable.

## Short Answer

Use both the frontend and VS Code.

The frontend is the governed control plane:

- log hypotheses
- lock experiment specs
- bind dataset and split signatures
- request review
- approve or reject experiments
- queue trainer-backed jobs
- start local workers
- inspect experiment queues
- inspect model packages and comparable-sales evidence
- create governance proposals

VS Code/CLI is the research and implementation workbench:

- run EDA
- inspect raw and processed data
- implement features and tests
- run targeted experiments that are too heavy or too exploratory for the UI
- generate markdown/CSV/JSON artifacts

The rule is simple: work can happen in VS Code, but evidence must land in the platform artifact folders.

## Required Artifact Locations

EDA artifacts:

- `reports/eda/eda_manifest_<tag>.json`
- `reports/eda/avm_eda_report_<tag>.md`
- `reports/eda/data_profile_<tag>.json`
- `reports/eda/segment_region_summary_<tag>.csv`
- `reports/eda/quarterly_market_trends_<tag>.csv`
- `reports/eda/feature_interaction_signals_<tag>.csv`
- `reports/eda/model_error_slices_<tag>.csv`
- `reports/eda/hypothesis_backlog_<tag>.md`

Experiment lifecycle artifacts:

- `reports/experiments/runs/<experiment_id>/experiment_spec.json`
- `reports/experiments/runs/<experiment_id>/dataset_snapshot.json`
- `reports/experiments/runs/<experiment_id>/run_manifest.json`
- `reports/experiments/runs/<experiment_id>/review_request.json`
- `reports/experiments/runs/<experiment_id>/review_decision.json`
- `reports/experiments/runs/<experiment_id>/job_manifest.json`
- `reports/experiments/runs/<experiment_id>/training_stdout.log`
- `reports/experiments/runs/<experiment_id>/training_stderr.log`
- `reports/experiments/runs/<experiment_id>/comparison_report.json`
- `reports/experiments/runs/<experiment_id>/audit_log.jsonl`

Model package artifacts:

- `models/packages/<package_id>/training_manifest.json`
- `models/packages/<package_id>/data_manifest.json`
- `models/packages/<package_id>/feature_contract.json`
- `models/packages/<package_id>/metrics.json`
- `models/packages/<package_id>/pre_comps_readiness.json`
- `models/packages/<package_id>/comps_manifest.json`
- `models/packages/<package_id>/selected_comps.csv`
- `models/packages/<package_id>/high_error_review_sample.csv`
- `models/packages/<package_id>/artifact_hashes.json`

## Senior DS Loop

1. Run EDA from current data.
2. Identify underperforming slices, sparse comp areas, non-stationary feature effects, and data-quality concerns.
3. Convert one finding into a frontend hypothesis.
4. Lock the spec in the Experiment Control Room.
5. Request review.
6. Approve only if the hypothesis has a clear model-risk boundary and dataset contract.
7. Queue training.
8. Inspect the package, metrics, comps, high-error review sample, and logs.
9. Compare champion/challenger on the same dataset snapshot and split signature.
10. Do not propose promotion until gates, evidence, and review artifacts support it.

## First EDA Command

```bash
python3 -m src.eda.real_estate_eda \
  --input-csv data/processed/nyc_sales_2019_2024_avm_training.csv \
  --predictions-csv reports/model/evaluation_predictions_v2_clean_baseline.csv \
  --tag initial_current_data
```

This produces a portfolio-ready EDA report and a hypothesis backlog under `reports/eda/`.

## What The EDA Must Look For

Real-estate AVM EDA is not only generic feature exploration. It must explicitly inspect:

- market segmentation by borough, neighborhood, property segment, and building class
- liquidity and sparse transaction cells
- comp coverage and comp dispersion
- temporal regime shifts and seasonality
- price-per-square-foot distribution by property type
- public-record missingness and imputation risk
- model errors by region, property type, price band, and comp coverage
- feature effects that change direction by location or property type
- likely unsupported cases that should trigger abstention

## Modeling Architecture Direction

The target architecture should be layered:

1. Comps-only baseline for transparency.
2. White-box baseline for interpretability.
3. Global gradient-boosted tree model for nonlinear public-record features.
4. Segmented router by property type and, when evidence supports it, geography.
5. Residual-over-comps model that predicts the correction from transparent comp evidence.
6. Confidence and abstention layer using comp coverage, comp recency, comp dispersion, feature completeness, out-of-distribution risk, and slice residuals.

Deep learning belongs in research only until the data supports it. It becomes credible when the project has listing text, photos, floor plans, neighborhood embeddings, or other high-dimensional signals.

## UI Connection

Current UI coverage:

- Experiment Control Room: hypothesis, locked spec, review, queue, worker start, registry.
- Artifact Explorer: model package, feature contract, model card, hashes, comparable-sales evidence.
- Governance: proposal and approval workflow.

Current limitation:

- EDA artifacts are generated and tracked in `reports/eda/`, but the UI does not yet have a dedicated EDA page. Until that page exists, EDA is run from VS Code/CLI and linked from experiment hypotheses.

## External Context

- IAAO AVM Standard: https://researchexchange.iaao.org/jptaa/vol15/iss2/5/
- Zillow Zestimate methodology: https://zillow.zendesk.com/hc/en-us/articles/4402325964563-How-is-the-Zestimate-calculated
- Fannie Mae property data collection and valuation modernization: https://www.fanniemae.com/research-and-insights/perspectives/advancing-collateral-valuation

