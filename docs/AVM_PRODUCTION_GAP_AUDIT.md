# S.P.E.C. NYC AVM and MLOps Production Gap Audit

Date: 2026-05-04  
Scope: Data science, AVM domain logic, MLOps governance, transparency, auditability, and demo/product readiness.

## Executive Positioning

S.P.E.C. NYC is not yet a production-grade AVM, but it is a strong production-oriented AVM case study. The repo already demonstrates better-than-typical portfolio discipline: contract-driven ETL, time-aware splits, leakage guards, segment scorecards, drift/performance monitors, MLflow-style lifecycle concepts, release validation, and a Next.js governance/monitoring surface.

The biggest gap is that the platform currently looks more production-grade than the valuation science underneath it. The operating system is ahead of the model. That is a fixable problem, and it is also the right story for a solo MLOps engineer: "I built the governed platform, then used it to expose where the model is not yet fit for high-stakes use."

The next version should be positioned as an NYC borough-level, public-data AVM research and governance platform, not a Zillow replacement. The quality bar should be: transparent, reproducible, point-in-time safe, calibrated, honest about confidence, and capable of rejecting properties when public data is not sufficient.

## Research Baseline

Production AVMs are not just tabular regression models. They are valuation systems that produce a value estimate at a valuation date, a confidence/uncertainty statement, coverage/hit-rate logic, comparable evidence, governance evidence, and audit controls.

Current real-estate AVM constraints to model around:

- Federal AVM quality-control standards are now active for covered mortgage use. The final interagency rule was published August 7, 2024 and became effective October 1, 2025. Covered institutions need policies, practices, procedures, and control systems for high confidence, data manipulation protection, conflict-of-interest controls, random sample testing/reviews, and nondiscrimination compliance.
- Regulators explicitly reject the idea that algorithmic valuation is automatically unbiased. AVMs can encode biased inputs, biased comparables, and biased design choices.
- Model risk management expectations are broader than accuracy. SR 11-7-style governance expects development controls, validation, implementation controls, use limitations, policies, issue management, and role separation.
- Industry AVM outputs commonly include estimated value, value range, confidence score, forecast standard deviation or equivalent uncertainty, and sometimes comparable sales.
- Industry AVM workflows often use cascades or model preference tables: model choice can depend on geography, property type, price range, confidence score, and forecast standard deviation.
- AVM accuracy is data-availability dependent. Zillow publicly distinguishes on-market and off-market accuracy and states that its Zestimate uses public records, MLS feeds, user-submitted data, property facts, location, and market trends. On March 6, 2026, Zillow reported national median error of 1.74% for on-market homes and 7.20% for off-market homes.
- Public NYC data is rich but incomplete. DOF rolling sales include neighborhood, building type, square footage, and sales data, while PLUTO exposes many parcel-level characteristics but also documents quality limitations, such as year-built values being approximate for many periods.

Sources are listed at the end of this report.

## Part 1: Current Platform Audit

### What Is Already Right

Data foundation:

- The repo uses NYC public sales data as a realistic bounded case study instead of synthetic Kaggle-only valuation.
- `src/etl.py` implements residential filtering, BBL/unit property IDs, deduplication, sales-history enrichment, segmentation, H3 indexing, distance-to-center features, temporal features, and imputation transparency.
- The project has a clear feature availability mindset. Current inference contracts explicitly ban target-derived fields like `sale_price` and `price_tier` in `src/inference.py`.
- There is a non-leaky `price_tier_proxy` path for routing, with train-fitted bins and inference-time fallback logic.

Modeling and evaluation:

- `src/model.py` uses a time split rather than a random-only split.
- The core model is an XGBoost tabular baseline, which is a sensible first production AVM model family.
- The system supports a global model and a segmented router with global fallback.
- Metrics include MdAPE, PPE10, R2, segment scorecards, price-tier scorecards, and temporal diagnostics.
- The challenger forensics reports correctly show an important senior-level behavior: do not promote a model because R2 improved if PPE10/MdAPE and major slices regress.

MLOps:

- The repo has MLflow run tracking concepts, model aliases, run cards, champion/challenger proposals, arena gates, release validation, drift monitoring, performance monitoring, and retrain policy outputs.
- The arena policy has real promotion rules: weighted segment MdAPE improvement, max major-segment PPE10 drop, major-segment floor, no new drift alerts, and no new fairness alerts.
- The project already has hypothesis docs and runbooks, which is exactly how real DS/MLOps teams avoid random retraining.

Product/demo:

- The dashboard has map-first valuation, governance, monitoring, artifacts, and copilot-oriented views.
- The UI exposes confidence, caveats, top drivers, model route, run ID, evidence paths, and monitoring state.

Verification run:

- Backend tests passed locally with the explicit suite: 54 tests, OK.
- Frontend typecheck did not run because local Node dependencies are missing: `tsc: command not found`.

### Critical Data Science Gaps

1. The current production artifact appears inconsistent with the current leakage policy.

`models/metrics_v1.json` records `price_tier` in `metadata.feature_columns`. In this project, `price_tier` is target-derived because it is assigned from within-segment `sale_price` quantiles in `src/etl.py`. Current inference code bans `price_tier`, so the active code and the saved production artifact are out of sync.

Impact: this is the most important current defect. It makes the artifact governance story weaker and may make current Python inference fail when loading the old model artifact.

Required fix: immediately retrain and re-register a clean baseline without `price_tier` in model features, regenerate metrics, SHAP, scorecards, monitoring artifacts, and release evidence.

2. ETL-level imputation can leak future distribution information.

`impute_missing_values` computes medians across the dataframe before modeling. If the full historical dataset is imputed before the time split, later-sale information influences earlier training rows and the holdout distribution.

Required fix: make imputation a train-fitted transformer inside the modeling pipeline, or make ETL imputation explicitly point-in-time. Keep imputation-source flags, but fit imputation statistics on the training window only.

3. The model target is too raw for real estate.

The current XGBoost model predicts raw `sale_price`. Real estate prices are skewed, heteroscedastic, and highly local. Production AVMs usually benefit from modeling log price, price per sqft with residual adjustment, or a hybrid hedonic/comps approach.

Required fix: benchmark these targets:

- `log1p(sale_price)` with inverse transform and residual correction.
- `log_price_per_sqft` plus sqft recomposition.
- Two-stage model: baseline hedonic price plus residual model.
- Quantile models for low/median/high instead of fixed percentage bands.

4. Current performance is not production-grade.

The current champion artifact reports:

- Overall PPE10: 0.3254.
- Overall MdAPE: 0.1637.
- Overall R2: 0.0281.
- ELEVATOR MdAPE: 0.2224, R2: -0.0931.
- WALKUP MdAPE: 0.2313.
- LUXURY MdAPE: 0.3335.

This is not a failure of the platform. It is an honest signal that the public-sales-only feature set is insufficient and/or the modeling formulation is underpowered for production valuation.

5. The current model lacks true comparable-sale logic.

AVMs are often expected to explain value through local market evidence. S.P.E.C. currently uses `h3_price_lag` and broad location features, but it does not yet build a proper comps engine.

Required fix:

- Build a comparable selection module with distance, recency, property type, sqft similarity, year-built/age similarity, unit count, borough/neighborhood, and sale validity filters.
- Compute comp-derived features: median comp price, median comp PPSF, comp count, comp recency, comp dispersion, nearest-comps distance, and local momentum.
- Expose the comps in the UI as evidence, not just as hidden features.

6. The platform needs abstention and hit-rate logic.

Production AVMs should not always return a confident value. They should know when public data is too sparse, stale, inconsistent, or out of distribution.

Required fix:

- Add `hit_status`: `hit`, `low_confidence_hit`, `no_hit`.
- Add `abstain_reason`: sparse comps, missing sqft, unknown property class, OOD features, stale market, high interval width, unusual transaction history.
- Track hit rate by borough, segment, price tier, and time.

7. Confidence is not yet calibrated.

The Next.js valuation response currently computes confidence from segment PPE10 and fixed completeness assumptions. Prediction intervals are fixed at +/-9%. That is good for a UI placeholder, but not a production AVM confidence system.

Required fix:

- Use residual-calibrated intervals by segment/borough/time.
- Add conformal prediction, ideally spatially or segment-aware.
- Track interval coverage: percent of actual sale prices inside 50%, 80%, and 90% intervals.
- Add forecast standard deviation or a MISMO-style common confidence score mapping as an output layer.

8. The current frontend valuation path is heuristic, not model-backed.

`web/src/bff/clients/canonicalValuationClient.ts` computes value from hard-coded borough PPSF, age multiplier, and unit multiplier. It links to model artifacts, but it does not call the Python model artifact or a model-serving API.

Required fix:

- Add a real model-serving boundary: `POST /api/v1/valuations/single` should call a Python/FastAPI service or a packaged inference function that loads the champion artifact.
- Return model route, feature vector hash, preprocessing version, prediction, interval, confidence, and explanation from the actual scorer.
- Keep the heuristic only as an explicit degraded fallback, and label it as fallback.

9. The dashboard SHAP summary is currently a static approximation.

`web/src/bff/clients/canonicalShapClient.ts` uses a hard-coded feature importance list and scales it by segment MdAPE. This is useful as a product scaffold, but it is not a true explainability service.

Required fix:

- Persist global SHAP tables as machine-readable CSV/JSON, not only PNG.
- Generate local SHAP values per valuation or approximate with cached background explainers.
- Separate "feature contribution" from "causal impact." The UI should support what-if simulation but clearly label it as sensitivity, not causal truth.

10. The feature set is still too thin for NYC valuation.

Current public-sales features are a start, but production-quality NYC valuation needs more sources:

- PLUTO: lot/building area, zoning, units, lot dimensions, land use, FAR, year altered, historic district.
- ACRIS/deeds: ownership transfers, mortgage/deed signals, transaction validity hints.
- DOB permits: renovation, alteration, new building, certificate of occupancy, open permits.
- HPD/DOB/OATH violations: condition and risk proxies.
- 311 and neighborhood condition signals.
- Transit access, parks, waterfront/flood risk, school-zone proxies where appropriate.
- Macro/time features: mortgage rates, borough-level market index, monthly volume/liquidity.
- If allowed: listings, days-on-market, list price, beds/baths, photos, descriptions.

For a solo public-data project, the realistic next data sources are PLUTO, DOB permits, HPD/DOB violations, ACRIS, subway distance, FEMA flood zones, and mortgage-rate time series.

### Critical MLOps and Governance Gaps

1. Release evidence and current code appear out of sync.

The production readiness report says all gates passed, but the current `validate_release.py` production checks include arena governance checks that are not shown in that older report. The saved model artifact also conflicts with current leakage rules.

Required fix: add a "current evidence freshness" gate. A release report should include code SHA, data version, model artifact hash, validation code version, and generated-at time, and should fail if any required artifact predates the current governance policy.

2. Governance checks are good but not yet real-estate-specific enough.

Current gates are generic ML quality gates plus segment stability. Real-estate AVM gates should add:

- Hit rate / coverage by borough and property segment.
- MdAPE, PPE5, PPE10, PPE20 by borough, segment, price tier, and time.
- COD-like dispersion of valuation ratios.
- PRD/PRB-style vertical equity checks across low/high value bands.
- Interval coverage and interval width.
- Overvaluation vs undervaluation asymmetry.
- Random sample review requirement.
- Low-confidence abstention quality.
- Non-market-sale contamination checks.
- Data freshness and stale-comps checks.

3. Fairness is currently too narrow.

The arena has `max_segment_ppe10_gap`, which is useful, but it is not enough for real estate. Fair housing risk is geographic and proxy-driven. Borough, neighborhood, building class, price tier, and public-record fields can encode socioeconomic and protected-class proxies.

Required fix:

- Do not train directly on protected class.
- Evaluate post-hoc using tract/neighborhood demographic proxies where legally and ethically appropriate for bias auditing.
- Track valuation ratio parity by geography and price bands.
- Add under/overvaluation asymmetry for historically undervalued neighborhoods.
- Add model-card language: intended use is research/demo and not a mortgage lending/appraisal substitute.

4. Monitoring is artifact-based, not production-observability-based.

Current drift/performance monitors read CSV artifacts. Production monitoring needs request-level telemetry, delayed ground truth joins, data quality alerts, model version dashboards, and SLOs.

Required fix:

- Log every prediction request with request ID, model version, feature schema version, input feature hash, confidence, route, interval width, latency, and response status.
- Join predictions to future observed sale prices when available.
- Monitor data drift, prediction drift, confidence drift, hit-rate drift, interval coverage, and segment performance.
- Add alert routing and incident runbooks.

5. Reproducibility needs artifact hashes and data versioning.

The repo has metrics artifacts, but no complete data lineage package.

Required fix:

- Store data source manifest: source URL, extract timestamp, row count, schema hash, file hash.
- Store feature-set version and transformation code hash.
- Store train/test split manifest.
- Store model artifact hash.
- Store environment lockfile. Python currently uses `requirements.txt`; a lock file or pinned constraints file would make the demo more reproducible.

6. Approval and audit should become first-class.

The governance UI is read-only, and that is fine for a no-auth demo. But production-grade MLOps needs immutable audit trails.

Required fix:

- Add a local SQLite/Postgres audit table for governance actions.
- Require approval reason, reviewer identity, proposal ID, old champion, new champion, artifact hashes, and rollback pointer.
- Add RBAC in the UI: viewer, model owner, risk reviewer, admin.
- Add a random sample review queue to satisfy the spirit of AVM quality-control standards.

## Part 2: Production-Grade Target State

### Realistic Target Architecture

For this project, "production-grade" should mean production-grade process and architecture, not Zillow-scale data coverage.

Target components:

1. Data ingestion:
   - DOF annualized/rolling sales.
   - PLUTO parcel/property attributes.
   - ACRIS deed/mortgage events.
   - DOB permits and job filings.
   - HPD/DOB/OATH violations.
   - Public geospatial features: subway distance, parks, flood zones, borough/neighborhood geometry.
   - Macro features: mortgage-rate and monthly market regime.

2. Data quality and point-in-time store:
   - Raw immutable extracts.
   - Canonical tables.
   - Point-in-time feature generation.
   - Sale validity/outlier labels.
   - Feature availability contracts.

3. Modeling:
   - White-box hedonic baseline.
   - Gradient boosting champion.
   - Comparable-sales model.
   - Repeat-sales / local price index model.
   - Quantile or conformal interval model.
   - Outlier/OOD detector.

4. Serving:
   - FastAPI scorer or model service.
   - Versioned preprocessing pipeline.
   - Request/response audit log.
   - Confidence and abstention policy.

5. MLOps:
   - Experiment tracking.
   - Model registry aliases.
   - Release gates.
   - Model card and validation report.
   - Monitoring with delayed labels.
   - Retrain decision policy.
   - Rollback workflow.

6. Product:
   - Map-based valuation workbench.
   - Comps and driver explanation.
   - Confidence/interval panel.
   - Governance gate board.
   - Monitoring and drift board.
   - Audit artifact explorer.

### Recommended Model Stack

Use multiple models, but do it conservatively. A good AVM stack for this repo is:

1. White-box hedonic model:
   - ElasticNet, GAM, or constrained linear model on log price.
   - Purpose: transparency, sanity check, model risk baseline.
   - Hiring-manager value: shows you understand regulated model validation and challenger baselines.

2. Gradient boosting champion:
   - XGBoost or LightGBM on log price or log PPSF.
   - Purpose: high-performing tabular nonlinear model.
   - Use monotonic constraints only where defensible, such as larger interior size generally increasing value within property type.

3. Comparable-sales model:
   - KNN/similarity search plus weighted median/trimmed mean comp price.
   - Purpose: domain-aligned explainability and fallback.
   - Also produces comp count, dispersion, and freshness for confidence.

4. Repeat-sales/local index model:
   - Estimate local market movement from repeat transactions or H3/neighborhood time indices.
   - Purpose: time adjustment for comps and market regime sensitivity.

5. Quantile/conformal interval layer:
   - Purpose: calibrated prediction intervals and confidence score.
   - Should be evaluated with interval coverage, not just interval width.

6. OOD/anomaly model:
   - Isolation forest, robust z-score rules, or density-based checks.
   - Purpose: flag unusual properties or likely non-market sales.

7. Optional deep learning:
   - Do not force deep learning into the core AVM unless you have data that warrants it.
   - Good uses: image/photo condition scoring, listing text embeddings, or learned neighborhood embeddings.
   - For a public-data project, a small embedding model for neighborhood/tract clusters is more realistic than a neural network price predictor.

### Where Supervised, Unsupervised, and Deep Learning Fit

Supervised learning:

- Main valuation models.
- Quantile models for intervals.
- Sale validity classifiers, if labels can be built.
- Confidence/hit-rate model predicting whether the AVM will be within 10%.

Unsupervised learning:

- Non-market sale anomaly detection.
- Neighborhood/tract clustering.
- Property-type submarket discovery.
- OOD detection for abstention.
- Drift clustering to explain why model quality changed.

Deep learning:

- Optional enhancement, not the core proof.
- Best used for condition/text/image signals if future listing/photo data is available.
- Could be showcased as a separate experiment: "Vision/text features are gated behind data availability and model-risk review."

### Domain-Specific Validation Gates

Add these gates before calling the platform production-grade:

Data gates:

- Required source manifests exist and hashes match.
- Sale validity filter report generated.
- Non-market sale contamination below threshold.
- Missingness by borough/segment below threshold or explicitly routed to abstain.
- No point-in-time leakage.
- No target-derived features in training or serving artifacts.

Model gates:

- Overall MdAPE and PPE10 pass minimum threshold.
- Borough/segment/price-tier slices pass floors.
- No major slice regresses more than allowed vs champion.
- Interval coverage is calibrated by segment.
- Hit rate is reported and above threshold.
- OOD/abstain rate is explainable.
- COD/PRD/PRB-style equity diagnostics pass or have documented remediation.
- Overvaluation/undervaluation asymmetry is bounded.

Governance gates:

- Model card exists and is complete.
- Independent validation checklist exists, even if "independent" means a separate documented review pass by you.
- Random sample testing queue generated.
- Approval reason is captured.
- Rollback pointer exists.
- All artifacts have hashes and are immutable for the release.

Monitoring gates:

- Prediction logs are being written.
- Drift alerts include root-cause slices.
- Ground-truth backfill job exists.
- Retrain decision uses both model age and observed performance decay.
- Dashboard clearly shows stale or missing evidence.

### How To Make It White-Box and Transparent

Do not try to make XGBoost fully white-box. Instead, make the platform white-box around it:

- Keep a white-box hedonic baseline next to the ML champion.
- Show top comparable sales and how much each supports the estimate.
- Show SHAP/local feature contributions for the champion.
- Show market trend adjustment separately from property-characteristic adjustment.
- Show confidence factors: comp count, comp recency, comp dispersion, input completeness, OOD score, interval width, segment calibration.
- Show "what would improve confidence": missing sqft, missing year built, sparse comps, stale market, unknown unit data.
- Show "not an appraisal" and "not for lending" caveats in the demo.

The best hiring-manager demo is not "here is one magic price." It is "here is the value, here is why the model believes it, here is where it is weak, here is the governance evidence, and here is when the system refuses to overstate confidence."

## Practical Solo-Dev Roadmap

### Phase 0: Artifact Integrity Fix

Goal: remove the immediate credibility gap.

- Retrain baseline with current feature contract, no `price_tier`.
- Regenerate metrics, predictions, SHAP, monitoring, retrain decision, and release report.
- Add a test that loads `models/model_v1.joblib` and validates its saved `feature_columns`.
- Add an artifact freshness gate comparing model metadata to current feature contract version.
- Install frontend dependencies or document bootstrap clearly so typecheck can run.

### Phase 1: Data Upgrade

Goal: make the AVM more domain-aware without needing proprietary MLS data.

- Add PLUTO join by BBL.
- Add DOB permit features: recent alteration, permit count, new construction, major work flags.
- Add violation/complaint features: open violations, severity counts, recency.
- Add transit/flood/geospatial features.
- Add point-in-time source manifests and data hashes.
- Replace simple `$10k` sale filter with a sale validity/outlier module.

### Phase 2: Model Upgrade

Goal: materially improve valuation science.

- Train white-box hedonic baseline.
- Train XGBoost on log target.
- Add comp engine and comp-derived features.
- Add rolling-origin backtests.
- Add calibrated interval model.
- Add hit/no-hit confidence policy.

### Phase 3: Governance Upgrade

Goal: make the MLOps platform real-estate-specific.

- Add AVM model card with intended use, limitations, data sources, protected-use caveats, confidence definition, and validation results.
- Add COD/PRD/PRB-style valuation-ratio diagnostics.
- Add fairness/proxy audit by geography and value band.
- Add random sample review artifacts.
- Add immutable promotion audit table.

### Phase 4: Product Demo Upgrade

Goal: make the demo compelling to hiring managers.

- Replace heuristic valuation endpoint with model-backed scoring.
- Show selected comps on the map and in a ranked comps table.
- Add confidence decomposition.
- Add interval coverage and hit-rate charts.
- Add governance evidence drawer.
- Add "model cannot confidently value this property" state.
- Add an experiment comparison page: champion vs challenger metrics, slices, fairness, calibration, and approval decision.

### Phase 5: Advanced ML Showcase

Goal: show breadth without overengineering the core.

- Add unsupervised submarket clusters.
- Add OOD detection and anomaly sale flags.
- Add optional deep-learning experiment only if you introduce text/image data.
- Add a "research not promoted" arena example showing why a complex model lost to a simpler champion.

## What Separates This From Zillow, HouseCanary, CoreLogic, and Lending-Grade Vendors

Data breadth:

- Large vendors have nationwide data, MLS/listing feeds, user-submitted facts, public records, assessor data, historical transaction depth, and sometimes proprietary condition/listing/valuation data.
- S.P.E.C. is currently public NYC sales plus engineered features. That is acceptable for a case study, but it limits accuracy and coverage.

Coverage:

- Production vendors optimize both accuracy and hit rate. They know where they can and cannot return a valuation.
- S.P.E.C. currently returns a valuation path in the UI but lacks real abstention and hit-rate accounting.

Confidence:

- Lending-grade AVMs use confidence scores, forecast standard deviation, value ranges, and policy thresholds.
- S.P.E.C. has confidence scaffolding but needs calibrated intervals and standardized confidence logic.

Governance:

- Covered mortgage workflows need AVM QC controls, random sample testing/reviews, nondiscrimination controls, and audit evidence.
- S.P.E.C. has the skeleton, but needs real-estate-specific gates and immutable audit trails.

Explainability:

- A strong AVM gives evidence: comps, drivers, confidence factors, and limitations.
- S.P.E.C. has SHAP artifacts and driver UI, but needs live local explanations and comp evidence.

Operational maturity:

- Production systems monitor serving traffic, label backfills, data freshness, source delays, calibration, drift, fairness, and incidents.
- S.P.E.C. monitors artifacts. That is a good start but not production observability.

## Recommended Narrative For Hiring Managers

Use this framing:

"This is an NYC public-data AVM and MLOps platform. It is not claiming Zillow-scale accuracy because I do not have Zillow-scale data. The goal is to show how I would build and govern a valuation system under real model-risk constraints: point-in-time data, leakage controls, champion/challenger releases, confidence calibration, fairness diagnostics, abstention, and audit evidence. The platform is deliberately transparent about model weaknesses and uses those weaknesses to drive the next DS experiments."

That is more credible than claiming a production-grade AVM from a limited public dataset.

## Priority Backlog

P0:

- Retrain clean artifact without `price_tier`.
- Add model artifact contract test.
- Replace frontend heuristic scorer with actual champion inference or label it degraded.
- Add source/data/model hash manifest to release reports.

P1:

- Add PLUTO features.
- Add log-target model.
- Build comp engine.
- Add calibrated intervals.
- Add hit/no-hit policy.

P2:

- Add COD/PRD/PRB-style diagnostics.
- Add fairness/proxy audit by geography/value band.
- Add immutable governance audit table.
- Add prediction logging and ground-truth backfill.

P3:

- Add unsupervised submarket clusters.
- Add OOD/anomaly sale module.
- Add deep-learning enrichment only with suitable image/text/listing data.

## Sources

- CFPB final AVM rule summary: https://www.consumerfinance.gov/rules-policy/final-rules/quality-control-standards-for-automated-valuation-models/
- GAO rule record with Federal Register publication and effective date: https://www.gao.gov/fedrules/211199
- CFPB blog on AVM accuracy/accountability and nondiscrimination: https://www.consumerfinance.gov/about-us/blog/cfpb-approves-rule-to-ensure-accuracy-and-accountability-in-the-use-of-ai-and-algorithms-in-home-appraisals/
- Federal Reserve/OCC SR 11-7 model risk management guidance: https://www.federalreserve.gov/bankinforeg/srletters/sr1107a1.pdf
- NIST AI Risk Management Framework overview: https://www.nist.gov/itl/ai-risk-management-framework
- Zillow Zestimate accuracy and methodology page, last updated March 6, 2026: https://www.zillow.com/zestimate/
- AVMetrics cascade glossary: https://www.avmetrics.net/AVM_glossary/cascade/
- Veros AVM cascade controls: https://www.veros.com/solutions/automated-valuation-solutions/avm-cascades
- MISMO AVM Common Confidence Score announcement via MBA: https://www.mba.org/news-and-research/newsroom/news/2025/08/21/mismo-releases-avm-common-confidence-score-standard-and-guidance-for-industry-use
- CoreLogic/Cotality AVM consumer assistance on value ranges, confidence scores, FSD, and comps: https://www.cotality.com/legal/avm-consumer-assistance
- NYC DOF rolling sales data: https://www.nyc.gov/site/finance/property/property-rolling-sales-data.page
- NYC PLUTO data dictionary: https://www.nyc.gov/assets/planning/download/pdf/data-maps/open-data/pluto_datadictionary.pdf
- IAAO Standard on Mass Appraisal, ratio uniformity, COD, PRD, PRB: https://www.iaao.org/wp-content/uploads/StandardOnMassAppraisal.pdf
