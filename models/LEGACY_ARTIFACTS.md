# Legacy Model Artifacts

The existing loose `v1` model artifacts are retained for historical comparison only.

They are not production-eligible under `docs/MODEL_ARTIFACT_CONTRACT.md` because they do not include a complete audit package and the saved `models/metrics_v1.json` metadata includes the target-derived `price_tier` feature.

Legacy files include:

- `models/model_v1.joblib`
- `models/metrics_v1.json`
- `reports/model/*_v1.*`
- `reports/arena/*`
- `reports/monitoring/*`

Next clean production candidates should be generated under:

```text
models/packages/<model_package_id>/
```
