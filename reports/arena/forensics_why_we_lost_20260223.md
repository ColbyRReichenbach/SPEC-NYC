# Challenger Forensics Memo (2026-02-23)

## Scope and Evidence
- Champion run: `34e917e198af4e58adb2097b8d9ca229` (`models/metrics_v1.json`)
- Reviewed challengers (promotion-eligible in arena artifacts):
  - `H-SEG-001` run `7f1cff9e88f54fe5a363248cc9a7da7a`
  - `H-SEG-TIER-001` run `d3bb60de10354f92be429f482fab2e4e`
  - `H-SEASON-001` run `6612686186654a3685df7ffca7ca3bde`
- Arena comparison source: `reports/arena/comparison_20260223T144123Z.csv`
- Note: recent `w6-validate-smoke` runs were excluded from forensic ranking because they are smoke validation artifacts (`dataset_version=w6_smoke`) and not promotion challengers.

## 1) Overall Comparison vs Champion

| Challenger | PPE10 | MdAPE | R2 | ΔPPE10 vs Champ | ΔMdAPE vs Champ | ΔR2 vs Champ |
|---|---:|---:|---:|---:|---:|---:|
| H-SEG-001 | 0.2213 | 0.2285 | 0.1701 | -0.1041 | 0.0649 | 0.1420 |
| H-SEG-TIER-001 | 0.2249 | 0.2292 | 0.1954 | -0.1005 | 0.0655 | 0.1673 |
| H-SEASON-001 | 0.2184 | 0.2521 | 0.3153 | -0.1070 | 0.0885 | 0.2872 |

## 2) Per-Segment Delta vs Champion (challenger - champion)

| Challenger | Segment | ΔPPE10 | ΔMdAPE | ΔR2 |
|---|---|---:|---:|---:|
| H-SEG-001 | ELEVATOR | -0.0529 | 0.0972 | 0.2076 |
| H-SEG-001 | SINGLE_FAMILY | -0.1825 | 0.0666 | -0.0778 |
| H-SEG-001 | SMALL_MULTI | -0.0371 | 0.0599 | -0.1005 |
| H-SEG-001 | WALKUP | -0.0171 | 0.0293 | -0.1192 |
| H-SEG-TIER-001 | ELEVATOR | -0.0413 | 0.0690 | 0.2482 |
| H-SEG-TIER-001 | SINGLE_FAMILY | -0.1895 | 0.0754 | -0.0014 |
| H-SEG-TIER-001 | SMALL_MULTI | -0.0315 | 0.0754 | -0.1813 |
| H-SEG-TIER-001 | WALKUP | -0.0025 | 0.0198 | -0.1922 |
| H-SEASON-001 | ELEVATOR | -0.0839 | 0.1632 | 0.3943 |
| H-SEASON-001 | SINGLE_FAMILY | -0.1463 | 0.0564 | 0.0163 |
| H-SEASON-001 | SMALL_MULTI | -0.0520 | 0.1146 | -0.1918 |
| H-SEASON-001 | WALKUP | -0.0626 | 0.0966 | -0.1580 |

## 3) Per-Tier Delta vs Champion (challenger - champion)

| Challenger | Tier | ΔPPE10 | ΔMdAPE | ΔR2 |
|---|---|---:|---:|---:|
| H-SEG-001 | core | -0.2079 | 0.0898 | -154.8757 |
| H-SEG-001 | entry | -0.0179 | 0.0849 | -101.3282 |
| H-SEG-001 | luxury | 0.0275 | -0.0754 | 0.3192 |
| H-SEG-001 | premium | -0.2180 | 0.0928 | -75.8267 |
| H-SEG-TIER-001 | core | -0.1962 | 0.0858 | -158.9934 |
| H-SEG-TIER-001 | entry | -0.0068 | 0.0471 | -123.2224 |
| H-SEG-TIER-001 | luxury | 0.0252 | -0.0645 | 0.3651 |
| H-SEG-TIER-001 | premium | -0.2222 | 0.1012 | -83.4521 |
| H-SEASON-001 | core | -0.1823 | 0.0987 | -53.9683 |
| H-SEASON-001 | entry | -0.0698 | 0.2066 | -48.3909 |
| H-SEASON-001 | luxury | 0.0268 | -0.0669 | 0.3991 |
| H-SEASON-001 | premium | -0.2086 | 0.0968 | -35.0012 |

## 4) Where Champion Still Dominates Exactly
- Overall: champion beats all challengers on both policy-critical metrics (higher PPE10, lower MdAPE).
- Segments: champion dominates all 4 major segments across all challengers (no segment where a challenger beats champion on both PPE10 and MdAPE).
- Tiers: champion dominates `core`, `entry`, and `premium` for all challengers; challengers only outperform in `luxury` tier.
- Policy implication: luxury-tier wins are too narrow to offset losses in high-volume slices, so every challenger fails arena gates.

## 5) Failure Classification by Challenger
### H-SEG-001
- Policy evidence: `gate_pass=False`, `weighted_segment_mdape_improvement=-0.4349`, `overall_ppe10_lift=-0.1041`, `max_major_segment_ppe10_drop=0.1825`, `min_major_segment_ppe10=0.1757`
- `feature gap`: Segment-only routing improves only luxury tier; core/entry/premium and all segments regress.
- `model tuning`: Objective/hyperparams increased R2 but worsened PPE10/MdAPE across policy slices.
- `policy-gate mismatch`: Failed all key arena quality gates (uplift, max drop, floor).
### H-SEG-TIER-001
- Policy evidence: `gate_pass=False`, `weighted_segment_mdape_improvement=-0.3858`, `overall_ppe10_lift=-0.1005`, `max_major_segment_ppe10_drop=0.1895`, `min_major_segment_ppe10=0.1832`
- `feature gap`: Non-leaky tier proxy helps luxury only; mass-market tiers still underfit.
- `split/eval design`: Route fragmentation (16 routes) likely increased variance for thinner slices.
- `model tuning`: Small PPE10 gain vs H-SEG-001 did not translate into policy-level improvements vs champion.
- `policy-gate mismatch`: Failed uplift/drop/floor gates despite no new drift/fairness alerts.
### H-SEASON-001
- Policy evidence: `gate_pass=False`, `weighted_segment_mdape_improvement=-0.6232`, `overall_ppe10_lift=-0.1070`, `max_major_segment_ppe10_drop=0.1463`, `min_major_segment_ppe10=0.1447`
- `feature gap`: Temporal features improved R2 but degraded MdAPE/PPE10 broadly, especially ELEVATOR and SMALL_MULTI.
- `model tuning`: Best-trial objective did not align with policy metrics in production slices.
- `policy-gate mismatch`: Largest weighted segment MdAPE regression among challengers.

## 6) Ranked Remediation Plan (No Retraining Yet)
| Priority | Action | Expected Impact | Effort |
|---|---|---|---|
| P0 | Re-align training objective to policy gates (primary optimize PPE10 + weighted segment MdAPE; constrain major-segment floor/drop during tuning). | high | medium |
| P1 | Route sparsity controls for segmented models: dynamic fallback to global when route train/test coverage is thin; enforce per-route holdout minimums. | high | medium |
| P2 | Tier feature improvement for mass market: v2 proxy focused on core/entry/premium discrimination (winsorized sqft/unit density + borough quantile anchors). | medium-high | medium-high |
| P3 | Time-x-slice evaluation hardening: rolling-window backtests with mandatory per-segment/per-tier acceptance thresholds before arena propose. | medium | low-medium |
| P4 | Data-quality audit on losing slices (core/premium tiers and ELEVATOR segment): outlier/duplicate/time-stamp consistency and leakage-safe freshness checks. | medium | medium |

## 7) Prioritized Experiment Queue
1. `H-OBJ-001` (new): policy-aligned objective/tuning (PPE10 + weighted segment MdAPE + segment floor constraints).
2. `H-ROUTE-002` (new): route coverage/fallback guardrails for segmented routing before enabling segment+tier by default.
3. `H-FEAT-003` (new): mass-market tier proxy v2 focused on core/entry/premium separability.
4. `H-EVAL-002` (new): rolling time-x-segment/tier acceptance protocol as pre-arena gate.
5. `H-DQ-001` (new): targeted data-quality forensic pass on losing slices.

## Why We Lost (Concise)
Challengers improved variance fit (R2) but lost on policy metrics (PPE10/MdAPE) in the slices that matter most. Gains were concentrated in luxury tier, while champion retained stronger accuracy across major segments and mass-market tiers; this created unavoidable arena gate failures (uplift negative, major-segment drop too high, segment floor violated).