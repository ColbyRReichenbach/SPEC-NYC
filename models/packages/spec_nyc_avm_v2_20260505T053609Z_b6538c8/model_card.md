# S.P.E.C. NYC AVM Candidate Model Card

Package ID: `spec_nyc_avm_v2_20260505T053609Z_b6538c8`

Model version: `v2`

## Intended Use

Candidate NYC borough-level AVM research model for governed valuation workflow demonstration.

## Prohibited Use

Not an appraisal, lending decision, tax assessment, insurance decision, or consumer-facing valuation product.

## Data Sources

Dataset version: `rows_96_date_2020-01-01_2025-03-15`.

## Training Window

Run started at `2026-05-05T05:36:09.180338Z`.

## Validation Window

Run finished at `2026-05-05T05:36:12.848003Z` with chronological holdout validation.

## Model Type

XGBRegressor

## Target

`sale_price` with target transform `none`.

## Features

Model features: `gross_square_feet, year_built, building_age, residential_units, total_units, distance_to_center_km, h3_prior_sale_count, h3_prior_median_price, h3_prior_median_ppsf, comp_count, comp_median_price, comp_median_ppsf, comp_weighted_estimate, comp_price_dispersion, comp_nearest_distance_km, comp_median_recency_days, comp_local_momentum, days_since_2019_start, month_sin, month_cos, borough, building_class, property_segment, neighborhood, rate_regime_bucket`. Router columns: `none`.

## Leakage Controls

Target-derived feature and router columns are blocked by package validation.

## Performance

Overall metrics: `{"coefficient_of_dispersion": 4.677924858594777, "mape": 0.09924575814722839, "mdape": 0.08870883201604571, "mean_valuation_ratio": 0.900847499345014, "median_valuation_ratio": 0.9112911679839543, "n": 20, "overvaluation_rate_10": 0.0, "overvaluation_rate_20": 0.0, "ppe10": 0.6, "ppe20": 0.95, "ppe5": 0.15, "price_related_bias": -0.45674969258818116, "price_related_differential": 1.0043639842073162, "r2": -1.229091096548888, "signed_pct_error_mean": -0.099152500654986, "signed_pct_error_median": -0.08870883201604571, "undervaluation_rate_10": 0.4, "undervaluation_rate_20": 0.05}`.

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
