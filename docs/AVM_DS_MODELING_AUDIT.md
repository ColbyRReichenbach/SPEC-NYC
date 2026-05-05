# S.P.E.C. NYC AVM Data Science Modeling Audit

Audit date: 2026-05-05

This audit covers the current AVM data science logic, how it differs from production-grade AVM practice, and how the next modeling rebuild should be shaped. It is intentionally realistic for a solo-dev NYC public-data case study: the goal is not to pretend we have Zillow or HouseCanary data density, but to show that the platform understands how AVMs should be modeled, validated, governed, and constrained.

Update: P0 pre-comps leakage controls from this audit have been implemented. See `docs/AVM_PRE_COMPS_READINESS_AUDIT.md` for the current readiness state before comparable-sales engine work.

Update: The first backend comparable-sales engine has been implemented in `src/features/comps.py`. It now produces strict as-of comp features and package evidence artifacts. Remaining comps-related work is confidence scoring, frontend comp display, and production-quality tuning of eligibility thresholds against larger validation evidence.

## Executive Finding

The MLOps shell is now ahead of the model. The current model is a governed, reproducible, auditable weak baseline. It is useful as a baseline artifact, but it is not a credible AVM champion.

The largest DS gaps are:

1. The target is modeled as raw sale price even though NYC sale prices are highly skewed and heteroscedastic.
2. The project does not yet have an as-of comparable-sales engine.
3. ETL-level imputation is fit on the full dataset before the train/test split, which leaks holdout distribution information.
4. The local `h3_price_lag` is holdout-safe but not fully point-in-time safe or cross-fitted.
5. The data cleaning layer does not yet produce a rigorous arm's-length sale-validity label.
6. Current validation uses one chronological holdout, not rolling-origin and spatial/segment stress tests.
7. Metrics are too generic for an AVM: we need valuation ratios, COD, PRD/PRB-style diagnostics, signed bias, interval coverage, and hit/no-hit.
8. There is no calibrated uncertainty, confidence model, or abstention policy.
9. Fairness/proxy auditing is documented but not executed.
10. The model is failing quality gates honestly, which is correct platform behavior.

## External Production AVM Baseline

Production AVMs are not just supervised regressors. They are valuation systems with data quality controls, market evidence, confidence logic, fairness controls, and recurring reviews.

Regulatory and standards anchors:

- The 2024 federal AVM final rule requires quality-control policies and systems for covered mortgage use cases that ensure confidence in estimates, protect against data manipulation, avoid conflicts of interest, require random sample testing/reviews, and comply with nondiscrimination laws.
- The CFPB identifies the AVM rule in Regulation Z section 1026.42(i), with implementation resources and compliance guide material.
- The IAAO AVM standard frames AVM work around statistically sufficient information, transparency, quality assurance, and appraisal principles.
- IAAO ratio-study practice gives useful AVM diagnostics even when this project is not doing tax assessment: valuation ratio, coefficient of dispersion, price-related differential, and price-related bias.

Market examples:

- Zillow describes Zestimate inputs as public, MLS, and user-submitted data, plus home facts, location, and market trends. Zillow also states that the Zestimate is not an appraisal and that accuracy depends on local data availability.
- Zillow's neural Zestimate announcement describes using deeper property history, sales transactions, tax assessments, public records, home details, location, and market trends, with neural models for scale.
- HouseCanary emphasizes dense normalized data, long history, frequent data refreshes, independent/pre-listing testing, fairness assessments, and transparency/API delivery.
- Recent AVM uncertainty research notes that tree ensembles and gradient boosted models are common for price prediction, but uncertainty intervals need spatial calibration because housing errors are spatially dependent.

Implication for S.P.E.C. NYC:

We should not claim national-AVM parity. We should show that we know what is missing, recreate the right patterns with public NYC data, and make the platform refuse promotion when model evidence is insufficient.

## Current Data Profile

Current processed dataset:

- Path: `data/processed/nyc_sales_2019_2024_avm_training.csv`
- Rows: 295,457
- Columns: 57
- Sale date range: 2019-01-01 to 2024-12-31
- Unique `property_id`: 256,430
- Borough coverage:
  - Manhattan: 80,141
  - Bronx: 23,736
  - Brooklyn: 74,715
  - Queens: 89,300
  - Staten Island: 27,565
- Segment coverage:
  - SINGLE_FAMILY: 128,267
  - ELEVATOR: 125,772
  - WALKUP: 26,007
  - SMALL_MULTI: 15,411

Observed quality issues:

- `sale_price` max is 598,155,755. That is a major outlier for a residential AVM target and needs sale-validity review before training.
- Price-per-square-foot p99 is 9,085.89, which strongly suggests either luxury extremes, sqft quality issues, bulk/atypical sales, or data errors.
- `gross_square_feet` has no missing values after ETL, but 52.6% of rows are imputed.
- `residential_units` and `total_units` are missing on 28.9% of rows.
- 5,547 rows duplicate on `bbl + sale_date + sale_price`, while `property_id + sale_date + sale_price` has no duplicates. This points to unit/property identity complexity, especially condos/coops.

## Current Model Evidence

Current candidate package:

- Package: `models/packages/spec_nyc_avm_v2_20260505T013717Z_b6538c8`
- Model: XGBRegressor
- Target: raw `sale_price`
- Transform: none
- Strategy: global
- Train rows: 236,365
- Holdout rows: 59,092
- Holdout window: 2023Q3 to 2024Q4
- Dataset version: `nyc_open_data_2019_2024_v2_bootstrap`

Overall holdout metrics:

| Metric | Value |
| --- | ---: |
| PPE5 | 9.3% |
| PPE10 | 18.5% |
| PPE20 | 35.3% |
| MdAPE | 31.0% |
| R2 | 0.305 |
| Median valuation ratio | 1.077 |
| IAAO-style COD | 93.6 |
| PRD-style diagnostic | 1.845 |
| PRB-style diagnostic | -1.225 |
| Overvaluation >20% | 41.2% |
| Undervaluation >20% | 23.5% |

Segment performance:

| Segment | n | PPE10 | MdAPE | Median Ratio | IAAO-style COD |
| --- | ---: | ---: | ---: | ---: | ---: |
| SINGLE_FAMILY | 25,148 | 27.5% | 19.4% | 0.982 | 62.5 |
| WALKUP | 5,143 | 14.3% | 39.0% | 1.096 | 147.5 |
| SMALL_MULTI | 3,018 | 13.6% | 40.8% | 1.128 | 118.0 |
| ELEVATOR | 25,783 | 11.1% | 50.7% | 1.278 | 93.5 |

Borough performance:

| Borough | n | PPE10 | MdAPE | Median Ratio | IAAO-style COD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Staten Island | 5,255 | 30.3% | 17.6% | 1.011 | 38.4 |
| Queens | 18,441 | 22.1% | 25.4% | 1.020 | 71.4 |
| Brooklyn | 14,481 | 18.8% | 28.7% | 1.029 | 102.0 |
| Bronx | 4,581 | 18.5% | 30.3% | 1.050 | 104.7 |
| Manhattan | 16,334 | 10.3% | 55.8% | 1.370 | 103.2 |

Price-tier diagnostics:

- Entry tier median ratio is 1.533, meaning the model materially overvalues lower-price rows.
- Luxury tier median ratio is 0.826, meaning the model materially undervalues high-price rows.
- This is vertical inequity behavior. It may reflect public-data limitations, target skew, missing condition/amenity features, or broken target/objective formulation.

## What The Current DS Logic Gets Right

These are real strengths and should be preserved:

- The current code blocks target-derived fields like `sale_price`, `price_tier`, and `predicted_price` from model features and routing.
- `price_tier_proxy` is explicitly non-target-derived and has tests proving target changes do not change the proxy.
- The model uses chronological holdout instead of random splitting.
- The training pipeline writes model packages with feature contracts, training manifests, validation reports, model cards, drift reports, explainability manifests, and artifact hashes.
- Segment, price-tier, temporal, missingness, and drift scorecards are generated.
- The governance layer correctly blocks weak candidates instead of creating a fake champion.
- The model card states limitations and prohibited use.

## Failed Or Missing Logic

### 1. Full-dataset ETL imputation leaks holdout distribution

`src/etl.py` imputes `gross_square_feet` and `year_built` before model splitting. Since the processed CSV already contains imputed values, the model's chronological holdout receives features whose imputation statistics were influenced by future holdout records.

Why this matters:

- 52.6% of rows have imputed square footage.
- Square footage is one of the most important valuation features.
- A production model needs imputation fit only on training windows and then applied forward.

Required fix:

- Move model-critical imputation into a fit/apply feature pipeline.
- Store imputation tables/statistics as artifacts.
- For each validation fold, fit imputation only on the training portion.
- Keep ETL imputation flags, but do not use full-dataset-imputed values as if they were raw facts.

### 2. Local price lag is not fully point-in-time safe

`add_h3_price_lag` computes an H3 median from the whole training split, then maps it into train and test. This is safe against the final holdout because test targets are not used. It is not fully production-safe because:

- Early training rows can receive medians influenced by later training sales.
- Optuna validation inside the training split receives `h3_price_lag` computed using validation targets.
- Each training row can influence its own encoded H3 median.

Required fix:

- Replace `h3_price_lag` with as-of rolling local features.
- For validation and training, generate out-of-fold or expanding-window encodings.
- Store the as-of feature spec and leakage proof in the feature contract.

### 3. The sale-validity model is too shallow

The current cleaning primarily filters `sale_price >= 10000`, residential categories, coordinates, and BBL. That is not enough for NYC public sales.

Missing sale-validity dimensions:

- Non-arm's-length transfers.
- Bulk portfolio/building transfers.
- Condo/co-op unit identity errors.
- Extreme PPSF rows.
- Very high prices inconsistent with class/size/segment.
- Repeated same-day BBL sales requiring unit-aware logic.
- Sales with imputed core features that should be downweighted or excluded from certain training roles.

Required fix:

- Add `sale_validity_status` with `valid_training_sale`, `review`, and `exclude_training`.
- Add `sale_validity_reasons` as an auditable list.
- Add anomaly detection as a review aid, not as an automatic truth label.
- Train only on valid/review-accepted transactions; score invalid rows separately for diagnostics.

### 4. Raw sale price target is the wrong first production target

The model directly predicts `sale_price`. For NYC, this forces one objective to handle $100k transfers, typical borough housing, luxury Manhattan units, and extreme outliers.

Observed symptom:

- Manhattan MdAPE is 55.8%.
- ELEVATOR MdAPE is 50.7%.
- Entry rows are overvalued while luxury rows are undervalued.

Required fix:

- Add target modes:
  - `log1p_sale_price`
  - `log_price_per_sqft`
  - `segment_residual_after_comps`
- Apply inverse-transform correction, such as residual smearing correction.
- Compare target modes on the exact same locked folds.

### 5. There is no real comparable-sales engine

Production AVMs usually anchor estimates to local market evidence. The current pipeline has `h3_price_lag`, but no selected comps, comp eligibility, comp dispersion, comp recency, or comp-adjusted valuation.

Required fix:

- Build an as-of comps module with eligibility rules:
  - sale occurs before valuation date
  - same borough
  - same or compatible segment/building class
  - distance/H3-ring threshold
  - sqft similarity
  - age/year-built similarity
  - unit-count similarity
  - valid sale only
- Persist selected comps for evaluated rows.
- Add comp-derived features:
  - comp count
  - median comp PPSF
  - weighted comp estimate
  - comp dispersion
  - nearest comp distance
  - median comp recency
  - local trend/momentum
- Use comps in both model features and user-facing evidence.

### 6. One chronological holdout is not enough

The current split is a simple chronological train/test split. That is better than random split, but it does not stress:

- Model stability across multiple origin dates.
- Locality transfer across neighborhoods or boroughs.
- Sparse segment behavior.
- Same-row champion/challenger comparisons across several folds.
- Future-period feature availability.

Required fix:

- Add rolling-origin validation, for example:
  - Train 2019-2021, validate 2022Q1-Q2
  - Train 2019-2022Q2, validate 2022Q3-Q4
  - Train 2019-2023Q2, validate 2023Q3-2024Q4
- Add spatial/submarket stress tests.
- Lock row IDs and feature snapshots for every fold.

### 7. Metrics need to become AVM metrics

Current metrics include PPE10, MdAPE, R2, segment/tier scorecards, and temporal scorecards. That is a good start but not enough.

Required additions:

- PPE5, PPE10, PPE20.
- MdAPE and MAPE as secondary only.
- Valuation ratio: `predicted_value / sale_price`.
- Median ratio by slice.
- COD-style dispersion by slice.
- PRD/PRB-style vertical equity diagnostics.
- Signed percentage error.
- Overvaluation and undervaluation rates.
- Interval coverage by slice.
- Hit/no-hit rate by slice.
- High-error sample review queue.

### 8. Drift logic overflags expected time features

The feature drift report flags many temporal features as alerts because the holdout period is later than training. That is expected for features like `days_since_2019_start`, `month_*`, and `rate_regime_bucket`.

Required fix:

- Separate expected temporal progression from suspicious covariate drift.
- Mark time-derived features as `calendar_progression` rather than generic drift.
- Keep drift alerts for non-time features and unexpected category/feature shifts.

### 9. SHAP is not enough for transparency

SHAP plots explain model mechanics, not causal property value drivers. Production-grade AVM transparency needs both:

- Model explanation: feature contributions, route, data quality, uncertainty.
- Market evidence: selected comps and comparable adjustments.

Required fix:

- Keep SHAP, but add local comp evidence.
- Show confidence and limitations based on comp density, missingness, OOD score, and interval width.
- Do not let SHAP substitute for valuation evidence.

### 10. No calibrated uncertainty or abstention

The current model card says calibrated conformal intervals are not implemented. That is a major production gap.

Required fix:

- Add residual/conformal intervals by segment/borough/price band.
- Track interval coverage at 50%, 80%, and 90%.
- Add a confidence score.
- Add `hit_status`:
  - `hit`
  - `low_confidence_hit`
  - `no_hit`
- Add `abstention_reason`, such as sparse comps, feature incompleteness, out-of-distribution property, stale market data, or unsupported segment.

### 11. Fairness/proxy audit is not executed

The model card says proxy audit is pending. The federal AVM rule makes nondiscrimination controls a core quality-control expectation for covered mortgage contexts.

Required fix:

- Evaluate valuation ratios and signed errors by borough, neighborhood, census tract, segment, and value band.
- Add caution around protected-class inference. This project should not infer protected class, but it can audit geographic and value-band proxies.
- Add promotion gates for material systematic over/undervaluation.
- Add random review samples for high-error and high-risk slices.

## Target Production-Grade Modeling Design

The realistic S.P.E.C. NYC target should be a governed AVM research system with three model layers.

### Layer 1: White-box benchmark

Purpose:

- Provide interpretability and sanity checks.
- Prevent tree/ensemble models from becoming the only truth source.
- Show hiring managers that the DS workflow respects baseline discipline.

Candidate models:

- ElasticNet on log price or log PPSF.
- GAM-style model if dependency choice is acceptable.
- Monotonic or constrained transformations for sqft, age, distance, and time.

Promotion role:

- Not necessarily champion, but required benchmark.
- Challenger must beat it on locked folds and major slices.

### Layer 2: Local comps engine

Purpose:

- Provide valuation evidence and domain credibility.
- Improve confidence scoring and abstention.
- Create transparent user-facing explanations.

Outputs:

- `comp_count`
- `weighted_comp_value`
- `median_comp_ppsf`
- `comp_dispersion`
- `nearest_comp_distance`
- `median_comp_recency_days`
- `local_market_momentum`
- persisted selected comps per evaluated/scored row

Promotion role:

- Required feature/evidence bundle for any credible AVM candidate.

### Layer 3: Predictive model or ensemble

Purpose:

- Learn nonlinear adjustments on top of property facts, time, locality, and comp features.

Candidate models:

- XGBoost/LightGBM-style gradient boosted trees.
- Segmented routers by property segment and enough-data submarkets.
- Ensemble of white-box baseline, comp estimate, and boosted residual model.
- Deep learning only if the repo adds suitable data, such as listing text, photos, or learned neighborhood embeddings.

Promotion role:

- Champion only if it passes absolute quality gates, slice floors, interval coverage, and fairness/proxy review.

## Next DS Implementation Order

### P0: Make the evaluation honest

1. Add `src/evaluate_avm.py` with AVM metrics:
   - PPE5/PPE10/PPE20
   - MdAPE
   - median valuation ratio
   - COD
   - PRD
   - signed percentage error
   - over/under valuation rates
2. Write these metrics into `metrics.json`, `validation_report.json`, and scorecards.
3. Add promotion gates using absolute thresholds and slice floors.
4. Add tests with known small arrays for COD/PRD/PPE behavior.

### P0: Fix leakage in feature construction

1. Stop treating ETL-imputed sqft/year-built as production-ready raw features.
2. Add train-fit/apply imputation artifacts for modeling.
3. Replace `h3_price_lag` with as-of rolling local features.
4. Make Optuna validation use only features fit before the validation window.

### P0: Add target modes

1. Add `--target-mode raw_price|log_price|log_ppsf`.
2. Implement inverse transform and correction.
3. Compare target modes on the same locked holdout rows.
4. Make `log_price` the default challenger unless evidence says otherwise.

### P1: Build comps

1. Add `src/features/comps.py`.
2. Generate as-of comp features for validation rows.
3. Persist selected comps for sample rows and high-error rows.
4. Add confidence inputs from comp count, comp dispersion, and recency.

### P1: Add rolling-origin validation

1. Create fold specs and row manifests.
2. Fit every preprocessing step per fold.
3. Compare white-box, comps-only, XGBoost, and ensemble candidates.
4. Require same-row challenger/champion comparison.

### P1: Add confidence and abstention

1. Add residual or conformal intervals.
2. Calibrate intervals by segment/borough/value band.
3. Add hit/no-hit status and abstention reasons.
4. Add interval coverage gates.

### P2: Public data expansion

Prioritize data sources that are public, defensible, and explainable:

1. PLUTO: lot area, building area, floors, land use, zoning, FAR, year altered.
2. DOB permits: alteration/new-build activity and recency.
3. HPD/DOB/OATH violations: quality/condition proxy with caution.
4. Transit/parks/flood/geospatial joins.
5. Borough/neighborhood/H3 rolling volume and price index proxies.

### P2: Fairness and sample review

1. Add geography/value-band proxy audit.
2. Add valuation ratio disparity gates.
3. Add random sample review artifacts.
4. Add high-error sample review artifacts.

## How We Should Talk About The Current Model

Correct positioning:

- "This is a governed weak baseline."
- "The platform refuses champion promotion because quality gates fail."
- "The next work is a DS rebuild around log targets, as-of comps, uncertainty, and AVM-specific ratio diagnostics."

Incorrect positioning:

- Do not call the current model production-grade.
- Do not approve it as champion just to make the UI look complete.
- Do not claim Zillow/HouseCanary parity.
- Do not treat SHAP as sufficient valuation transparency.

## Source Notes

- Federal Reserve Board, multi-agency AVM final rule press release, July 17, 2024: https://www.federalreserve.gov/newsevents/pressreleases/bcreg20240717a.htm
- CFPB AVM quality-control standards resource page: https://www.consumerfinance.gov/compliance/compliance-resources/mortgage-resources/quality-control-standards-for-automated-valuation-models/
- IAAO, "Standard on automated valuation models (AVMs)", Journal of Property Tax Assessment & Administration, revised July 2018: https://researchexchange.iaao.org/jptaa/vol15/iss2/5/
- IAAO, "Standard on Ratio Studies", 2013: https://www.iaao.org/wp-content/uploads/Standard_on_Ratio_Studies.pdf
- Zillow, "What is a Zestimate?": https://mortgage.zillow.com/zestimate/
- Zillow, "Zillow Launches New Neural Zestimate, Yielding Major Accuracy Gains", June 15, 2021: https://www.zillow.com/news/zillow-launches-new-neural-zestimate-yielding-major-accuracy-gains/
- HouseCanary, "Our AVM": https://www.housecanary.com/resources/our-avm
- Hjort et al., "Uncertainty quantification in automated valuation models with spatially weighted conformal prediction", arXiv, revised Jan. 30, 2025: https://arxiv.org/abs/2312.06531
