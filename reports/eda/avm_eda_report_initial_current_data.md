# S.P.E.C. NYC Senior DS EDA Report

Generated: 2026-05-05T05:55:11.566305Z

## Workflow Answer

The frontend owns governed experiment lifecycle actions: hypothesis logging, review, queueing, worker start, governance, and package inspection. VS Code/CLI owns heavier EDA and feature research. This EDA job writes immutable artifacts under `reports/eda/`, so the analysis is still captured by the platform rather than living only in a notebook.

## Dataset Profile

- Rows: 295,457
- Sale window: 2019-01-01 to 2024-12-31
- Median sale price: $775,000
- Median PPSF: $471
- Unique properties: 256430

## Segment And Region Structure

| borough | property_segment | n | median_sale_price | median_ppsf | median_sqft | median_building_age |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ELEVATOR | 65554 | 1225000.000 | 928.108 | 1848.000 | 65.000 |
| 4 | SINGLE_FAMILY | 51686 | 799000.000 | 493.167 | 1596.000 | 90.000 |
| 3 | SINGLE_FAMILY | 35061 | 995000.000 | 521.620 | 2100.000 | 100.000 |
| 4 | ELEVATOR | 27466 | 410000.000 | 238.095 | 1848.000 | 64.000 |
| 5 | SINGLE_FAMILY | 25962 | 625000.000 | 396.952 | 1531.500 | 52.000 |
| 3 | ELEVATOR | 24161 | 710000.000 | 586.722 | 1321.000 | 62.000 |
| 2 | SINGLE_FAMILY | 14290 | 650000.000 | 323.661 | 1995.000 | 90.000 |
| 3 | WALKUP | 8694 | 995000.000 | 471.461 | 1848.000 | 97.000 |

## Real-Estate Specific Modeling Issues

- Location effects are not globally stationary. Distance-to-center, density, and unit-count signals can change sign by borough and property type.
- Property type should be treated as a model boundary, not only a feature. Single-family, small multifamily, walkup, elevator, condo/coop-like records have different comp logic.
- Public records miss condition, renovation, views, floor level, layout, concessions, and listing demand. The model must expose confidence and abstention, not force a value.
- Comparable-sales evidence should be both a model feature layer and a reviewer-facing explanation layer.

## Feature Interaction Signals

| scope | borough | property_segment | feature | n | spearman_corr_log_ppsf | direction |
| --- | --- | --- | --- | --- | --- | --- |
| borough_segment | 4 | ELEVATOR | gross_square_feet | 27466 | -0.798 | negative |
| borough_segment | 3 | SMALL_MULTI | gross_square_feet | 6799 | -0.721 | negative |
| borough_segment | 3 | ELEVATOR | gross_square_feet | 24161 | -0.695 | negative |
| borough_segment | 5 | ELEVATOR | gross_square_feet | 960 | -0.688 | negative |
| borough_segment | 5 | WALKUP | total_units | 341 | -0.617 | negative |
| borough_segment | 5 | WALKUP | residential_units | 341 | -0.617 | negative |
| segment | ALL | ELEVATOR | gross_square_feet | 125772 | -0.568 | negative |
| borough_segment | 1 | ELEVATOR | gross_square_feet | 65554 | -0.557 | negative |
| borough_segment | 3 | ELEVATOR | distance_to_center_km | 24161 | -0.538 | negative |
| borough | 4 | ALL | gross_square_feet | 89300 | -0.516 | negative |

## Model Underperformance Signals

Prediction artifact analyzed: `reports/model/evaluation_predictions_v2_clean_baseline.csv`.

| slice_type | slice_name | n | mdape | ppe10 | median_signed_pct_error | overvaluation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| neighborhood | HILLCREST | 82 | 1.549 | 0.037 | 1.549 | 0.951 |
| neighborhood | HARLEM-EAST | 145 | 1.425 | 0.062 | 1.425 | 0.848 |
| neighborhood | CHINATOWN | 48 | 1.235 | 0.000 | 1.235 | 0.833 |
| neighborhood | BEDFORD PARK/NORWOOD | 131 | 1.045 | 0.069 | 1.045 | 0.802 |
| neighborhood | HARLEM-CENTRAL | 494 | 0.998 | 0.077 | 0.998 | 0.800 |
| neighborhood | LONG ISLAND CITY | 627 | 0.996 | 0.037 | -0.919 | 0.062 |
| neighborhood | INWOOD | 104 | 0.991 | 0.029 | 0.991 | 0.885 |
| neighborhood | RIVERDALE | 804 | 0.899 | 0.086 | 0.195 | 0.524 |
| neighborhood | WASHINGTON HEIGHTS LOWER | 88 | 0.899 | 0.068 | 0.889 | 0.682 |
| neighborhood | CLINTON | 310 | 0.807 | 0.065 | 0.415 | 0.610 |

## Architecture Recommendation

Use a layered AVM architecture rather than one monolithic model:

1. Comps-only transparent baseline.
2. Global gradient-boosted tree model with point-in-time features.
3. Segmented router by property type and, when data supports it, geography.
4. Residual-over-comps model that predicts the correction from transparent market evidence.
5. Confidence, hit/no-hit, and abstention layer from comp coverage, dispersion, feature completeness, and slice residuals.

Deep learning should stay experimental until richer modalities exist, such as listing text, images, floor plans, or neighborhood embeddings.

## Hypothesis Backlog

These are candidate hypotheses for the governed experiment UI. Each should be converted into a locked spec before training.

## Underperforming Slices

- Investigate `neighborhood=HILLCREST`: n=82, MdAPE=1.549, PPE10=0.037. Candidate action: segment-specific residual calibration or abstention rule.
- Investigate `neighborhood=HARLEM-EAST`: n=145, MdAPE=1.425, PPE10=0.062. Candidate action: segment-specific residual calibration or abstention rule.
- Investigate `neighborhood=CHINATOWN`: n=48, MdAPE=1.235, PPE10=0.000. Candidate action: segment-specific residual calibration or abstention rule.
- Investigate `neighborhood=BEDFORD PARK/NORWOOD`: n=131, MdAPE=1.045, PPE10=0.069. Candidate action: segment-specific residual calibration or abstention rule.
- Investigate `neighborhood=HARLEM-CENTRAL`: n=494, MdAPE=0.998, PPE10=0.077. Candidate action: segment-specific residual calibration or abstention rule.
- Investigate `neighborhood=LONG ISLAND CITY`: n=627, MdAPE=0.996, PPE10=0.037. Candidate action: segment-specific residual calibration or abstention rule.
- Investigate `neighborhood=INWOOD`: n=104, MdAPE=0.991, PPE10=0.029. Candidate action: segment-specific residual calibration or abstention rule.
- Investigate `neighborhood=RIVERDALE`: n=804, MdAPE=0.899, PPE10=0.086. Candidate action: segment-specific residual calibration or abstention rule.

## Non-Stationary Feature Effects

- Test interaction or segmented treatment for `building_age` because observed direction changes across flat, negative, positive.
- Test interaction or segmented treatment for `distance_to_center_km` because observed direction changes across flat, negative, positive.
- Test interaction or segmented treatment for `gross_square_feet` because observed direction changes across negative, positive.
- Test interaction or segmented treatment for `residential_units` because observed direction changes across flat, negative, positive.
- Test interaction or segmented treatment for `total_units` because observed direction changes across flat, negative, positive.

## Sparse Segment/Region Cells

- Add hit/no-hit or fallback policy for borough `2` / segment `SMALL_MULTI` because observed row count is 186.
- Add hit/no-hit or fallback policy for borough `5` / segment `SMALL_MULTI` because observed row count is 42.

## Architecture Hypotheses

- Compare comps-only estimate, global XGBoost, segmented router, and residual-over-comps candidate on identical rows.
- Add confidence/abstention using comp count, comp recency, comp dispersion, feature missingness, and slice residuals.
- Add PLUTO only after current error and comp coverage artifacts show which property facts are missing.

## External Standards And Industry Context

- IAAO describes AVM standards as principles and best practices for developing and using AVMs for real property valuation: https://researchexchange.iaao.org/jptaa/vol15/iss2/5/
- Zillow says its Zestimate uses home facts, location, market trends, comparable homes, prior sales, and public/off-market records: https://zillow.zendesk.com/hc/en-us/articles/4402325964563-How-is-the-Zestimate-calculated
- Fannie Mae emphasizes standardized, objective property data collection for AVM, appraisal, market analysis, and compliance use cases: https://www.fanniemae.com/research-and-insights/perspectives/advancing-collateral-valuation

## Latest Quarterly Market Trend Sample

| period | borough | property_segment | n | median_ppsf | ppsf_qoq_change |
| --- | --- | --- | --- | --- | --- |
| 2024Q4 | 2 | SINGLE_FAMILY | 525 | 369.940 | 0.006 |
| 2024Q4 | 3 | SMALL_MULTI | 244 | 729.985 | 0.266 |
| 2024Q4 | 2 | SMALL_MULTI | 4 | 196.110 | -0.068 |
| 2024Q4 | 3 | SINGLE_FAMILY | 1369 | 567.376 | -0.008 |
| 2024Q4 | 5 | SMALL_MULTI | 1 | 186.667 | -0.466 |
| 2024Q4 | 2 | WALKUP | 57 | 134.033 | -0.205 |
| 2024Q4 | 5 | SINGLE_FAMILY | 951 | 450.274 | -0.002 |
| 2024Q4 | 5 | WALKUP | 29 | 255.405 | 0.367 |
