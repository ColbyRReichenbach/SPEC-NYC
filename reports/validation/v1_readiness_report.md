# S.P.E.C. NYC V1 Readiness Report

- Mode: `production`
- Started (UTC): `2026-05-05T00:03:14.711579`
- Finished (UTC): `2026-05-05T00:03:34.260959`
- Duration: `19.55s`

## Checks

| Check | Evidence Type | Status | Detail | Log |
|---|---|---|---|---|
| unit_tests | production | pass | exit=0; OK | reports/validation/logs/unit_tests.log |
| production_data_evidence | production | pass | found 1 production ETL markdown + 1 csv reports | - |
| production_model_evidence | production | fail | model package contract failed: models/packages/spec_nyc_avm_v2_20260504T235549Z_b6538c8 - [release_decision] release_decision.decision must be approved for production eligibility (release_decision.json) | - |
| production_product_evidence | production | pass | all required artifacts present | - |
| production_ops_evidence | production | pass | all required artifacts present | - |
| streamlit_app_production | production | pass | app process stayed healthy for startup window | reports/validation/logs/streamlit_app_production.log |
| arena_governance_production | production | fail | champion alias not set for spec-nyc-avm (Registered model alias champion not found.) | - |
| release_tag | release | pass | tagging skipped (--tag-release not set) | - |

## Gates

| Gate | Status | Failed Checks | Missing Checks |
|---|---|---|---|
| Gate A (Data) | done | - | - |
| Gate B (Model) | blocked | production_model_evidence, arena_governance_production | - |
| Gate C (Product) | done | - | - |
| Gate D (Ops) | done | - | - |
| Gate E (Release) | blocked | - | - |

## Artifacts

- JSON payload: `reports/validation/v1_readiness_report.json`
- Command logs: `reports/validation/logs`