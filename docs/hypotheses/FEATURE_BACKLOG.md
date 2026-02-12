# Feature Backlog (Model Inputs and Candidates)

Purpose:
- Show what features we use now.
- Prioritize what to add next.
- Tie feature work to hypothesis IDs.

---

## 1) Current Model Features (As Of Now)

Training features currently used by `src/model.py`:

Numeric:
1. `gross_square_feet`
2. `year_built`
3. `building_age`
4. `residential_units`
5. `total_units`
6. `distance_to_center_km`
7. `h3_price_lag`

Categorical:
1. `borough`
2. `building_class`
3. `property_segment`
4. `neighborhood`

Important leakage note:
- `price_tier` is **not** used as a model feature now.
- `price_tier` in ETL is target-derived from `sale_price`, so it is reserved for analysis/slicing.

---

## 2) Existing ETL Features Not Yet Used in Model

These already exist in the dataset and can be tested safely:
1. `sale_month`
2. `sale_quarter`
3. `price_per_sqft` (use with caution due target linkage in historical rows)
4. `sale_sequence`
5. `days_since_last_sale`
6. `price_change_pct` (careful: historical availability logic required)
7. `sqft_imputed`, `sqft_imputation_level`
8. `year_built_imputed`

Notes:
- For production inference, ensure feature is available at prediction time.
- If a feature depends on prior sale data, define fallback behavior for first-time or sparse-history properties.

---

## 3) High-Value Feature Candidates (Prioritized)

## P0: Transit accessibility (NYC-specific)
1. `distance_to_nearest_subway_station`
2. `subway_stations_within_0_5mi`
3. `subway_stations_within_1_0mi`
4. `nearest_station_line_count` (optional)

Why:
- Transit is a major price signal in NYC.

Hypothesis:
- `H-FEAT-001-subway-proximity`

Dependencies:
- External station dataset join.

---

## P1: Stronger time/regime features
1. `days_since_2019_start` (trend term)
2. `month_sin`, `month_cos` (cyclical seasonality)
3. `rate_regime_bucket` (if macro source added)

Why:
- Real estate pricing is regime-sensitive and seasonal.

Hypothesis:
- `H-SEASON-001`

---

## P1: Local market intensity
1. `h3_recent_sales_count_90d`
2. `h3_recent_median_ppsf_90d`
3. `borough_recent_turnover_rate`

Why:
- Captures local liquidity and momentum.

Hypothesis:
- `H-LIQ-001`

---

## P2: Building and zoning enrichment
1. Building age buckets
2. Tax class encoding refinement
3. Optional PLUTO/zoning enrichments

Why:
- Can improve structural comparability.

Hypothesis:
- `H-STRUCT-001`

---

## 4) Segment+Tier Routing Clarification

`router_mode=segment_plus_tier` now requires:
1. `property_segment`
2. `price_tier_proxy` (non-leaky)

Current state:
- `price_tier_proxy` is not yet engineered in ETL by default.
- Do not use target-derived `price_tier` for routing.

Recommended next hypothesis:
- `H-FEAT-002-price-tier-proxy` to define and validate a non-leaky proxy tier.

---

## 5) Feature Acceptance Criteria

A new feature is accepted when all are true:
1. Available at prediction time (no future/target leakage).
2. Improves target metric under arena policy gates.
3. Does not introduce fairness/drift regressions.
4. Added to dataset/change documentation with versioned feature set.
