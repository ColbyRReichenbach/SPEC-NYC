# AVM EDA Hypothesis Backlog

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
