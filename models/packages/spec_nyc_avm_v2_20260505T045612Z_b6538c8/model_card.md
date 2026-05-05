# S.P.E.C. NYC AVM Candidate Model Card

Package ID: `spec_nyc_avm_v2_20260505T045612Z_b6538c8`

Model version: `v2`

## Intended Use

Candidate NYC borough-level AVM research model for governed valuation workflow demonstration.

## Prohibited Use

Not an appraisal, lending decision, tax assessment, insurance decision, or consumer-facing valuation product.

## Data Sources

Dataset version: `precomps_smoke_fixture`.

## Training Window

Run started at `2026-05-05T04:56:12.074647Z`.

## Validation Window

Run finished at `2026-05-05T04:56:18.805135Z` with chronological holdout validation.

## Model Type

XGBRegressor

## Target

`sale_price` with target transform `none`.

## Features

Model features: `gross_square_feet, year_built, building_age, residential_units, total_units, distance_to_center_km, h3_prior_sale_count, h3_prior_median_price, h3_prior_median_ppsf, days_since_2019_start, month_sin, month_cos, borough, building_class, property_segment, neighborhood, rate_regime_bucket`. Router columns: `none`.

## Leakage Controls

Target-derived feature and router columns are blocked by package validation.

## Performance

Overall metrics: `{"coefficient_of_dispersion": 6.420965515519662, "mape": 0.06422405669358085, "mdape": 0.052306767702228, "mean_valuation_ratio": 0.9838914745964136, "median_valuation_ratio": 0.9764029932882123, "n": 28, "overvaluation_rate_10": 0.10714285714285714, "overvaluation_rate_20": 0.03571428571428571, "ppe10": 0.75, "ppe20": 0.9642857142857143, "ppe5": 0.42857142857142855, "price_related_bias": -0.5651353384565313, "price_related_differential": 1.008017876301086, "r2": 0.2971513088994283, "signed_pct_error_mean": -0.016108525403586652, "signed_pct_error_median": -0.023597006711787653, "undervaluation_rate_10": 0.14285714285714285, "undervaluation_rate_20": 0.0}`.

## Slice Performance

See `slice_scorecard.csv` and `validation_report.json`.

## Confidence and Intervals

Candidate package does not yet contain calibrated conformal intervals.

## Fairness and Proxy Audit

Proxy audit is pending; review geography, value band, and segment valuation-ratio gaps before promotion.

## Limitations

Candidate package generated from available project data only. Not approved for production use until release_decision.decision is updated by governance workflow. Public NYC data may omit condition, renovation quality, listing media, concessions, and private transaction context.

## Known Failure Modes

Sparse local comps, unusual condition, major renovations, non-market sales, and missing public-record fields.

## Monitoring Plan

Monitor feature drift, prediction drift, hit rate, interval coverage, stale data, and segment decay.

## Rollback Plan

Use the previous approved champion package once release governance records a rollback pointer.
