# S.P.E.C. NYC V1 Readiness Report

- Mode: `production`
- Started (UTC): `2026-02-23T14:42:44.987600`
- Finished (UTC): `2026-02-23T14:43:04.311774`
- Duration: `19.32s`

## Checks

| Check | Evidence Type | Status | Detail | Log |
|---|---|---|---|---|
| unit_tests | production | pass | exit=0; OK | reports/validation/logs/unit_tests.log |
| production_data_evidence | production | pass | found 6 production ETL markdown + 6 csv reports | - |
| production_model_evidence | production | pass | production model metadata validated (train_rows=236365, threshold=5000) | - |
| production_product_evidence | production | pass | all required artifacts present | - |
| production_ops_evidence | production | pass | all required artifacts present | - |
| streamlit_app_production | production | pass | app process stayed healthy for startup window | reports/validation/logs/streamlit_app_production.log |
| arena_governance_production | production | fail | latest proposal is not approved (status=no_winner) | - |
| release_tag | release | pass | tagging skipped (--tag-release not set) | - |

## Gates

| Gate | Status | Failed Checks | Missing Checks |
|---|---|---|---|
| Gate A (Data) | done | - | - |
| Gate B (Model) | blocked | arena_governance_production | - |
| Gate C (Product) | done | - | - |
| Gate D (Ops) | done | - | - |
| Gate E (Release) | blocked | - | - |

## Artifacts

- JSON payload: `reports/validation/v1_readiness_hseason001_production.json`
- Command logs: `reports/validation/logs`