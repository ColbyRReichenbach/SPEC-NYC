# H-SEASON-001 Temporal Evaluation Memo (2026-02-23)

- Hypothesis: `H-SEASON-001`
- Challenger run: `7a7a754a94d5435f9131f6ddd0dcfb32`
- Champion run: `34e917e198af4e58adb2097b8d9ca229`
- Feature set version: `fs_seasonality_regime_v1`
- Dataset version: `ds_hseason001_train_20260223`
- Recommendation: **REWORK**

## 1) Headline Outcome (Challenger vs Champion)
- Overall PPE10: `0.2186` vs `0.3254` (delta `-0.1068`)
- Overall MdAPE: `0.2524` vs `0.1637` (delta `+0.0887`)
- Overall R2: `0.3063` vs `0.0281` (delta `+0.2782`)

## 2) Time-Based Stability
- Quarter window: `2023Q3` to `2024Q4`
- MdAPE std: challenger `0.0099` vs champion `0.0046` (delta `+0.0052`)
- PPE10 std: challenger `0.0080` vs champion `0.0065` (delta `+0.0014`)
- MdAPE range delta: `+0.0177`
- PPE10 range delta: `+0.0019`

Major-segment temporal stability deltas (challenger - champion):
- `ELEVATOR`: MdAPE std delta `+0.0375`, PPE10 std delta `+0.0010`
- `SINGLE_FAMILY`: MdAPE std delta `+0.0047`, PPE10 std delta `+0.0135`
- `SMALL_MULTI`: MdAPE std delta `+0.0149`, PPE10 std delta `-0.0022`
- `WALKUP`: MdAPE std delta `+0.0022`, PPE10 std delta `-0.0006`

## 3) Slice Performance
Major segments:
- `ELEVATOR`: PPE10 delta `-0.1066`, MdAPE delta `+0.2137`
- `SINGLE_FAMILY`: PPE10 delta `-0.1268`, MdAPE delta `+0.0455`
- `SMALL_MULTI`: PPE10 delta `-0.0639`, MdAPE delta `+0.1230`
- `WALKUP`: PPE10 delta `-0.0352`, MdAPE delta `+0.0471`

Price tiers:
- `core`: PPE10 delta `-0.1919`, MdAPE delta `+0.1014`
- `entry`: PPE10 delta `-0.0751`, MdAPE delta `+0.2422`
- `luxury`: PPE10 delta `+0.0303`, MdAPE delta `-0.0742`
- `premium`: PPE10 delta `-0.1981`, MdAPE delta `+0.0923`

## 4) Arena Gate Position
- Proposal: `reports/arena/proposal_8a33c69e69d9.json`
- Status: `no_winner`
- Candidate gate row:
```json
{
  "run_id": "7a7a754a94d5435f9131f6ddd0dcfb32",
  "model_version": "v1",
  "score": -0.72,
  "gate_pass": false,
  "weighted_segment_mdape_improvement": -0.7000811446896945,
  "overall_ppe10_lift": -0.10679956677722871,
  "max_major_segment_ppe10_drop": 0.12676622572213903,
  "min_major_segment_ppe10": 0.12190202846837063,
  "drift_alert_delta": 0,
  "fairness_alert_delta": 0
}
```

Gate viability: `not viable`.
Because gates are not viable, no promotion package is prepared.

## 5) Artifact Paths
- Summary JSON: `reports/arena/hseason001_compare_summary_20260223_t1555.json`
- Run card (challenger): `reports/arena/run_card_7a7a754a94d5435f9131f6ddd0dcfb32.md`
- Metrics (challenger): `models/metrics_v1_hseason001_20260223_t1555.json`
- Segment scorecard (challenger): `reports/model/segment_scorecard_v1_hseason001_20260223_t1555.csv`
- Temporal scorecard (challenger): `reports/model/temporal_scorecard_v1_hseason001_20260223_t1555.csv`
- Temporal scorecard recomputed (champion): `reports/model/temporal_scorecard_v1_baseline_recomputed_20260223_t1555.csv`
- Temporal scorecard recomputed (challenger): `reports/model/temporal_scorecard_v1_hseason001_20260223_t1555_recomputed.csv`
- Missingness diagnostics (challenger): `reports/model/feature_missingness_v1_hseason001_20260223_t1555.csv`
- Drift diagnostics (challenger): `reports/model/feature_drift_segment_time_v1_hseason001_20260223_t1555.csv`
- Arena comparison CSV: `reports/arena/comparison_20260223T155837Z.csv`
- Arena proposal MD: `reports/arena/proposal_8a33c69e69d9.md`
