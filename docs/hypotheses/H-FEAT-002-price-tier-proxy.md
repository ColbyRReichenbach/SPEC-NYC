# H-FEAT-002: Non-Leaky price_tier_proxy

Date: 2026-02-23
Owner: colby

## Objective
Provide a routing-safe `price_tier_proxy` for `router_mode=segment_plus_tier` using only inference-available inputs.

## Construction Strategy
`price_tier_proxy` is derived from a non-target proxy score:

- `gross_square_feet` (log-scaled size signal)
- `distance_to_center_km` (closer => higher signal)
- `building_age` (newer => higher signal)
- `total_units` / `residential_units` (light density signal)
- `borough` (coarse prior)

No use of:
- `sale_price` (target)
- `price_tier` (target-derived)

## Tier Assignment
1. Compute proxy score per row.
2. Fit per-segment q25/q50/q75 thresholds on training split only.
3. Apply fitted thresholds to train/test/inference rows.
4. Use global threshold fallback for sparse/unknown segments.
5. Use `core` fallback when score inputs are missing.

## Leakage Guardrails
- Training blocks any routing on `price_tier`.
- Inference blocks artifacts configured with router column `price_tier`.
- Tests verify proxy stability when target columns are perturbed.

## Inference Availability
- If incoming frame lacks `price_tier_proxy`, inference derives it from artifact-stored bins and non-target signals.

