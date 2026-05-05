# S.P.E.C. NYC AVM Candidate Model Card

Package ID: `spec_nyc_avm_v2_20260504T235549Z_b6538c8`

Model version: `v2`

## Intended Use

Candidate NYC borough-level AVM research model for governed valuation workflow demonstration.

## Prohibited Use

Not an appraisal, lending decision, tax assessment, insurance decision, or consumer-facing valuation product.

## Data Sources

Dataset version: `nyc_open_data_2019_2024_v2_bootstrap`.

## Training Window

Run started at `2026-05-04T23:55:49.491780Z`.

## Validation Window

Run finished at `2026-05-04T23:56:07.559870Z` with chronological holdout validation.

## Model Type

XGBRegressor

## Target

`sale_price` with target transform `none`.

## Features

Model features: `gross_square_feet, year_built, building_age, residential_units, total_units, distance_to_center_km, h3_price_lag, days_since_2019_start, month_sin, month_cos, borough, building_class, property_segment, neighborhood, rate_regime_bucket`. Router columns: `none`.

## Leakage Controls

Target-derived feature and router columns are blocked by package validation.

## Performance

Overall metrics: `{"mdape": 0.31313148437500005, "n": 59092, "ppe10": 0.1708691531848643, "r2": 0.3138433722772701}`.

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
