# S.P.E.C. NYC Production AVM Backlog

Date: 2026-05-04  
Purpose: Convert the production gap audit into concrete implementation work across MLOps, data science, and frontend/product.

## Senior Recommendation

Start with MLOps, but do not pause everything for a full codebase rewrite.

The right sequence is:

1. Freeze the current model artifacts as legacy evidence.
2. Define the production artifact standard and governance contracts.
3. Build tests around those contracts.
4. Scrap/retrain the modeling artifacts from a clean DS baseline.
5. Wire the frontend to real model-backed artifacts and scoring.

Do not begin by trying to prove every existing code path is correct. That will turn into a broad refactor with unclear finish criteria. Instead, define the non-negotiable production invariants first: no target leakage, point-in-time data, reproducible configs, immutable artifact hashes, model card, validation report, approval record, rollback pointer, and audit log. Then test those invariants aggressively.

The current platform should become an artifact factory. Every training run should produce a complete evidence package that a model-risk reviewer could inspect without asking the developer what happened.

## Operating Principles

- Treat all current model artifacts as disposable until regenerated under the new contracts.
- Keep code changes small and gate-driven.
- Prefer explicit manifests over implicit file naming.
- Every model release must answer: what data, what features, what code, what params, what metrics, what slices, what risks, who approved, and how to roll back.
- The UI should never imply more confidence than the evidence supports.
- "Production-grade" here means governed, transparent, auditable, reproducible, and honest about public-data limitations.

## Current Implementation Status

Completed on 2026-05-04:

- Added the model artifact contract: `docs/MODEL_ARTIFACT_CONTRACT.md`.
- Added audit-grade package validation: `src/mlops/artifact_contract.py`.
- Added legacy marker documentation: `models/LEGACY_ARTIFACTS.md`.
- Updated production release validation to require a contract-valid model package at `models/packages/current` by default.
- Added tests for valid packages, target-derived feature rejection, artifact hash mismatch, stable SHA-256 hashing, missing package rejection, and train-row threshold enforcement.
- Added separate router-column validation so segmented routing cannot use target-derived fields such as `price_tier`.
- Added inference allowlist enforcement so model packages cannot self-declare unsupported feature columns as safe.
- Added candidate package generation from training outputs: `src/mlops/package_builder.py`.
- Updated `src/model.py` so future training runs write candidate packages under `models/packages/<model_package_id>/`.
- Bootstrapped NYC Open Data annualized sales: 498,666 raw records from 2019-01-01 through 2024-12-31.
- Wrote transformed training snapshot: `data/processed/nyc_sales_2019_2024_avm_training.csv` with 295,457 residential market-like sales and 256,430 unique properties.
- Trained clean v2 global baseline from the processed NYC snapshot and generated candidate package `models/packages/spec_nyc_avm_v2_20260504T235549Z_b6538c8`.
- Candidate package validates structurally with pending release allowed; production validation still fails until governance approval is recorded.
- Rebuilt the Next.js frontend into a governance-first AVM workbench that reads model package, ETL, and release artifacts.
- Added a collapsible product shell and an experiment control room for hypothesis logging, governed preflight runs, and local experiment manifests under `reports/experiments/`.
- Added a locked experiment registry contract: each new preflight writes `experiment_spec.json`, `dataset_snapshot.json`, `run_manifest.json`, `comparison_report.json`, and `audit_log.jsonl` under `reports/experiments/runs/<experiment_id>/`.
- Added registry API support through `/api/v1/experiments/registry` and upgraded `/api/v1/experiments/preflight` to create the locked artifact bundle.
- Added champion/challenger comparison preconditions to experiment artifacts: comparison is blocked until a challenger has scored the same dataset snapshot and split signature.
- Added governed experiment lifecycle APIs:
  - `/api/v1/experiments/[experimentId]/review-request`
  - `/api/v1/experiments/[experimentId]/review`
  - `/api/v1/experiments/[experimentId]/queue`
  - `/api/v1/experiments/[experimentId]/worker`
- Added lifecycle artifacts for review and execution: `review_request.json`, `review_decision.json`, `job_manifest.json`, `training_stdout.log`, and `training_stderr.log`.
- Added a real local worker entry point at `src/experiments/worker.py` that reads queued job manifests, calls the existing `src.model` trainer, updates run status, writes logs, and produces a champion/challenger comparison report from the generated model package.
- Updated the experiment control room so a DS user can create a locked spec, queue it for review, approve/reject it, queue a trainer-backed job, inspect review/job artifacts, and see review/training/tracked queues.
- Added a local release proposal and champion registry workflow:
  - `reports/governance/proposals/<proposal_id>/release_proposal.json`
  - `reports/governance/proposals/<proposal_id>/approval_decision.json`
  - `reports/governance/proposals/<proposal_id>/audit_log.jsonl`
  - `reports/governance/audit_log.jsonl`
  - `models/packages/aliases/spec-nyc-avm.json`
- Added governance APIs for proposal creation, approval, and rejection:
  - `/api/v1/governance/proposals`
  - `/api/v1/governance/proposals/[proposalId]/approve`
  - `/api/v1/governance/proposals/[proposalId]/reject`
- Added proposal quality gates so a challenger must pass same-dataset comparison and configured absolute AVM quality thresholds before it can become champion.
- Added governance UI controls for proposal generation, approval/rejection with required reason, champion registry inspection, rollback pointer inspection, and proposal gate evidence.
- Expanded CI to discover and run the full backend unittest suite.
- Verified frontend CI after installing dependencies: brand guard, lint, typecheck, and contract tests pass.
- Verified the frontend with configured Playwright e2e tests for desktop, mobile, collapsed-sidebar, preflight, review approval, training queue, governance proposal creation, and governance rejection workflows.

Current blockers:

- Raw and processed training data now exist locally but are intentionally gitignored due size. Reproducibility depends on rerunning connector + ETL, not committing the CSV.
- Production data evidence now passes against the v2 ETL report.
- No production-approved model package exists yet. The completed smoke challenger was correctly blocked because it missed configured absolute quality thresholds: minimum overall PPE10 and maximum overall MdAPE.
- Local champion registry exists, but no champion alias is set for `spec-nyc-avm` until a candidate passes proposal gates and is approved.
- `npm install` reports 17 dependency vulnerabilities, including a Next.js security warning; dependency upgrade work is still needed.
- The new frontend intentionally blocks price generation until an approved model package and scoring bridge exist.
- DS modeling audit is now documented in `docs/AVM_DS_MODELING_AUDIT.md`; the next implementation work should follow that audit before any champion/scoring work.

Not completed in this pass:

- DS model rebuilding has a clean v2 baseline, but it is not production quality: latest completed challenger PPE10 is 18.5% and MdAPE is 31.0% on the holdout. This is acceptable as a governed baseline, not as a deployable AVM.
- Frontend model-backed valuation is not complete because there is no approved model package or serving bridge to call.
- Experiment training is now routeable through a queued job manifest and local worker. A real worker smoke run completed for `exp_20260505T013702_0e9d360f`, producing challenger package `models/packages/spec_nyc_avm_v2_20260505T013717Z_b6538c8`, non-empty stdout/stderr logs, and a passed same-dataset comparison report. Additional queued jobs can be started from the dashboard or with `python3 -m src.experiments.worker --repo-root . --experiment-id <id> --once`.
- Dataset snapshots are locked by package data hash and split signature. Row-level split ID materialization is still a future enhancement before this can fully guarantee row-by-row challenger/champion parity outside the current artifact contract.
- Release approval workflow is implemented for local champion aliasing. Promotion to a serving/scoring runtime is still not complete; the next step is to make the scorer load only `models/packages/aliases/spec-nyc-avm.json`.

Completed on 2026-05-05:

- Added a governed DS experiment workflow runbook at `docs/DS_EXPERIMENT_WORKFLOW.md`.
- Added a reproducible senior-DS EDA artifact generator at `src/eda/real_estate_eda.py`.
- Added a deterministic as-of comparable-sales engine at `src/features/comps.py`.
- Added point-in-time comp eligibility:
  - valid historical sales only
  - strict `comp.sale_date < valuation.sale_date`
  - same borough
  - same or compatible property segment
  - distance, square-footage, age, and unit-count similarity rules
  - explicit sparse-market fallback policy
- Added comp-derived model features:
  - `comp_count`
  - `comp_median_price`
  - `comp_median_ppsf`
  - `comp_weighted_estimate`
  - `comp_price_dispersion`
  - `comp_nearest_distance_km`
  - `comp_median_recency_days`
  - `comp_local_momentum`
- Added selected comparable-sales evidence artifacts:
  - `comps_manifest.json`
  - `selected_comps.csv`
  - `high_error_review_sample.csv`
  - `high_error_selected_comps.csv`
- Wired comps features into the training feature contract, inference allowlist, package feature descriptions, and candidate model packages.
- Added a package-level comparable-sales evidence panel to the artifact explorer.
- Added unit tests proving the comps engine excludes same-day sales, holdout targets, invalid/review sales, wrong-borough sales, and unsupported target-derived behavior.
- Added a frontend EDA & Hypothesis Lab at `/eda` that reads `reports/eda/*`, shows DS findings, and can convert an EDA backlog item into a locked experiment preflight through `/api/v1/experiments/preflight`.
- Added a dynamic `/artifact-viewer` route and EDA artifact index so notebooks, reports, CSVs, JSON, and HTML artifacts generated after new EDA runs can be opened in-app from backend-discovered paths.
- Added Playwright coverage for the EDA lab across desktop/mobile, including sidebar collapse behavior, failed-response/console-error checks, hypothesis preflight creation, and page-level horizontal overflow checks.

## Phase Order

### Phase 0: Reset and Contract Definition

- Mark existing `models/model_v1.joblib`, `models/metrics_v1.json`, and related reports as legacy artifacts.
- Create a new artifact version namespace, for example `v2`.
- Define required release artifacts.
- Define required training-run artifacts.
- Define model feature contract schema.
- Define validation report schema.
- Define governance audit schema.
- Add tests that fail if a production model artifact contains target-derived fields.

Exit criteria:

- The repo has a documented artifact contract.
- Tests can validate whether an artifact package is production-eligible.
- Existing leaky/stale artifacts are no longer treated as production evidence.

### Phase 1: MLOps Foundation

- Implement reproducible artifact generation.
- Implement immutable manifests and hashes.
- Implement audit-grade model cards.
- Implement release package validation.
- Implement governance decision records.
- Implement prediction logging schema, even before serving is fully production.

Exit criteria:

- A model run produces a complete, inspectable evidence bundle.
- A release check can pass/fail based on that bundle.
- A reviewer can trace model output to data, code, features, params, metrics, and approval.

### Phase 2: Clean Modeling Baseline

- Retrain from scratch with the new artifact standard.
- Remove target-derived fields from model features.
- Move train-fitted imputation/preprocessing into the model pipeline.
- Add log-target and comps-derived baselines.
- Add calibrated intervals and hit/no-hit policy.

Exit criteria:

- New clean champion candidate exists.
- All model artifacts pass feature and leakage checks.
- Metrics are honest and segmented.
- Confidence is calibrated from residual evidence, not hard-coded UI logic.

### Phase 3: Frontend Evidence Integration

- Replace heuristic valuation with model-backed scoring.
- Add artifact explorer.
- Add model card view.
- Add feature contract view.
- Add confidence decomposition.
- Add comps and SHAP evidence views.
- Add governance audit trail UI.

Exit criteria:

- A user can run or inspect a valuation and see model version, features, confidence, drivers, comps, caveats, and evidence paths.
- Governance and monitoring views are backed by real artifacts.

## MLOps TODO

### MLOps P0: Artifact Contract and Legacy Reset

- [x] Create `docs/MODEL_ARTIFACT_CONTRACT.md`.
- [x] Define a required model package layout:
  - [x] `model.joblib`
  - [x] `metrics.json`
  - [x] `model_card.md`
  - [x] `training_manifest.json`
  - [x] `data_manifest.json`
  - [x] `feature_contract.json`
  - [x] `validation_report.json`
  - [x] `slice_scorecard.csv`
  - [x] `temporal_scorecard.csv`
  - [x] `drift_report.json`
  - [x] `explainability_manifest.json`
  - [x] `release_decision.json`
  - [x] `artifact_hashes.json`
- [x] Add a `legacy` marker for current v1 artifacts.
- [x] Stop production release checks from accepting legacy artifacts as current evidence.
- [x] Add a `model_package_id` convention, for example `spec_nyc_avm_v2_<timestamp>_<git_sha>`.
- [x] Add a required `feature_contract_version`.
- [x] Add a required `dataset_version`.
- [x] Add a required training code identifier through `training_manifest.git_sha`.
- [x] Add a required model artifact hash through `artifact_hashes.json`.
- [x] Add a required `data_snapshot_sha256`.

### MLOps P0: Leakage and Feature-Availability Gates

- [x] Add a production artifact validator module, for example `src/mlops/artifact_contract.py`.
- [x] Validate that `feature_columns` contain no forbidden target-derived fields:
  - [x] `sale_price`
  - [x] `price_tier`
  - [x] `predicted_price`
  - [x] `abs_pct_error`
  - [x] any post-sale outcome fields
- [x] Validate router columns separately from model feature columns.
- [x] Validate every feature has an inference availability declaration.
- [x] Validate every feature has a point-in-time availability declaration.
- [x] Validate every feature has owner, description, dtype, null policy, and source.
- [x] Fail release if a feature is not covered by the inference availability allowlist.
- [x] Fail release if feature contract and model artifact disagree.
- [x] Add tests that load model packages and enforce the feature contract.

### MLOps P0: Reproducibility Manifests

- [x] Create `training_manifest.json` generation.
- [x] Include:
  - [x] command used
  - [x] git SHA
  - [x] Python version
  - [x] package versions
  - [x] random seed
  - [x] train/test split logic
  - [x] model class
  - [x] hyperparameters
  - [x] target transform
  - [x] preprocessing steps
  - [x] optimization objective
  - [x] run start/end time
- [x] Create `data_manifest.json`.
- [x] Include:
  - [x] source name
  - [x] source URL or local path
  - [x] extract timestamp
  - [x] raw row count
  - [x] post-filter row count
  - [x] schema hash
  - [x] file hash
  - [x] min/max sale date
  - [x] data freshness
  - [x] known limitations
- [x] Create `artifact_hashes.json`.
- [x] Hash all files in the model package.
- [x] Fail release if any referenced hash is missing.

### MLOps P0: Audit-Grade Model Card

- [x] Replace `reports/releases/model_card_template.md` with a production-contract model card template and generate package model cards from training.
- [ ] Include:
  - [x] intended use
  - [x] prohibited use
  - [x] not-an-appraisal disclaimer
  - [x] data sources
  - [x] training window
  - [x] validation window
  - [x] model type
  - [x] target definition
  - [x] feature list
  - [x] excluded/leakage-blocked fields
  - [x] performance summary
  - [ ] borough metrics
  - [x] segment metrics
  - [x] price-band metrics
  - [x] temporal metrics
  - [ ] confidence calibration
  - [ ] interval coverage
  - [ ] fairness/proxy audit summary
  - [x] limitations
  - [x] known failure modes
  - [x] monitoring plan
  - [x] rollback plan
- [x] Add tests that required model card sections are present.

### MLOps P1: Governance and Release Workflow

- [ ] Expand `config/arena_policy.yaml` for AVM-specific gates:
  - [x] minimum overall MdAPE
  - [x] minimum overall PPE10
  - [ ] minimum borough PPE10
  - [ ] minimum segment PPE10
  - [ ] maximum major-slice regression
  - [ ] maximum interval miscalibration
  - [ ] minimum hit rate
  - [ ] maximum no-hit rate by major segment
  - [ ] maximum overvaluation bias
  - [ ] maximum undervaluation bias
  - [ ] COD threshold
  - [ ] PRD/PRB threshold
  - [ ] no target leakage
  - [ ] no stale data
  - [ ] no missing model card
  - [x] no missing rollback pointer
- [x] Add generated `release_decision.json`.
- [x] Include:
  - [x] proposal ID
  - [x] previous champion package ID
  - [x] candidate package ID
  - [x] gate results
  - [x] approver
  - [x] approval/rejection reason
  - [x] timestamp
  - [x] rollback package ID
  - [x] artifact hashes
- [x] Add append-only `reports/governance/audit_log.jsonl`.
- [ ] Add append-only audit events:
  - [ ] training run created
  - [ ] candidate registered
  - [x] proposal generated
  - [x] proposal approved
  - [x] proposal rejected
  - [x] champion changed
  - [ ] rollback executed
- [ ] Add tests that audit records cannot be malformed.

### MLOps P1: Random Sample Review

- [ ] Add random sample review generation for each candidate.
- [ ] Sample across borough, segment, price band, and confidence band.
- [ ] Persist `random_review_sample.csv`.
- [ ] Persist `random_review_summary.md`.
- [ ] Include actual sale price, predicted value, interval, error, confidence, route, and top drivers.
- [ ] Add reviewer fields:
  - [ ] reviewer
  - [ ] review status
  - [ ] risk notes
  - [ ] action required
- [ ] Gate promotion on review artifact existence.

### MLOps P1: Monitoring and Prediction Logging

- [ ] Create prediction log schema.
- [ ] Log:
  - [ ] request ID
  - [ ] valuation ID
  - [ ] timestamp
  - [ ] model package ID
  - [ ] model alias
  - [ ] feature contract version
  - [ ] feature vector hash
  - [ ] route
  - [ ] predicted value
  - [ ] interval low/high
  - [ ] confidence score
  - [ ] hit/no-hit status
  - [ ] abstention reason
  - [ ] latency
  - [ ] error status
- [ ] Add delayed ground-truth join design.
- [ ] Monitor:
  - [ ] input drift
  - [ ] prediction drift
  - [ ] confidence drift
  - [ ] hit-rate drift
  - [ ] interval coverage
  - [ ] segment performance decay
  - [ ] stale source data

### MLOps P2: CI and Test Expansion

- [x] Fix test discovery or document explicit test command.
- [x] Add CI target for all backend tests, not only selected smoke tests.
- [x] Add artifact contract tests.
- [ ] Add model package load tests.
- [x] Add release validator tests.
- [ ] Add audit log tests.
- [ ] Add dashboard contract tests for governance/monitoring payloads.
- [ ] Add a bootstrap check for missing Node dependencies.
- [ ] Pin Python dependencies or add a lock/constraints file.

## Data Science TODO

### DS Audit Fix Checklist

Source audit: `docs/AVM_DS_MODELING_AUDIT.md`.

#### P0: Governed EDA Workflow

- [x] Document frontend vs VS Code responsibilities for governed DS work.
- [x] Add reproducible EDA artifact generation under `reports/eda/`.
- [x] Generate data profile, segment/region summaries, quarterly trends, feature-effect diagnostics, model-error slices, and hypothesis backlog.
- [x] Add a dedicated frontend EDA artifact page.

#### P0: Evaluation and Governance Evidence

- [x] Convert DS audit findings into an executable fix checklist.
- [x] Add AVM-specific metric helpers.
- [x] Emit PPE5, PPE10, PPE20, MdAPE, MAPE, valuation ratio, COD, PRD, PRB-style bias, signed error, and over/under valuation rates.
- [x] Add optional interval coverage and hit/no-hit rate metrics when those fields exist.
- [x] Add targeted tests for AVM metric math and scorecard output.
- [ ] Add absolute gate thresholds for the new ratio/bias diagnostics.
- [ ] Add slice-floor gates for borough, segment, price tier, and high-imputation slices.
- [ ] Add high-error review sample artifacts.

#### P0: Leakage and Point-in-Time Features

- [x] Move model-critical sqft/year-built imputation out of full-dataset ETL and into train-fit/apply feature artifacts.
- [x] Store imputation policy/statistics in every new model package through `pre_comps_readiness.json`.
- [x] Replace current `h3_price_lag` with strict as-of H3 local market features.
- [x] Make Optuna/internal validation use features fit only before the validation window.
- [x] Add row/fold feature snapshot manifests that prove challenger and champion used the same rows.
- [x] Add feature availability manifest for all pre-comps model features.

#### P0: Target and Objective

- [ ] Add `raw_price`, `log_price`, and `log_ppsf` target modes.
- [ ] Add inverse-transform correction for log targets.
- [ ] Compare targets on identical locked rows/folds.
- [ ] Add composite objective across MdAPE, PPE10/PPE20, slice floors, and bias diagnostics.

#### P1: Sale Validity and Comps

- [x] Add `sale_validity_status` and `sale_validity_reasons`.
- [x] Flag likely non-arm's-length, duplicate/unit-identity, extreme PPSF, extreme sale-price, rapid resale, and ETL-imputed core-feature rows.
- [x] Build an as-of comparable-sales engine.
- [x] Persist selected comps for evaluated rows and high-error review samples.
- [x] Add comp-count, comp-recency, comp-dispersion, and weighted comp estimate features.

#### P1: Validation, Confidence, and Fairness

- [ ] Add rolling-origin validation.
- [ ] Add residual/conformal intervals.
- [ ] Add confidence score, hit/no-hit, and abstention reasons.
- [ ] Add interval coverage gates by slice.
- [ ] Add geography/value-band proxy fairness audit.
- [ ] Add random sample review and high-risk sample review artifacts.

### DS P0: Clean Baseline From Scratch

- [x] Treat current model artifacts as legacy.
- [x] Retrain a clean baseline without `price_tier`.
- [x] Move target-derived tier labels to evaluation-only fields.
- [x] Ensure `price_tier_proxy` is the only tier-like routing feature allowed.
- [x] Ensure imputation/preprocessing statistics are fit on training data only.
  - Current status: ETL-imputed sqft/year-built values are reset to missing before model fitting; sklearn imputers fit on training windows only.
- [ ] Add a simple white-box baseline:
  - [ ] linear model
  - [ ] ElasticNet
  - [ ] or GAM-style model if dependency choice is acceptable
- [ ] Add XGBoost baseline using current clean features.
- [ ] Compare white-box baseline vs XGBoost.
- [x] Generate model card and validation report for the clean v2 baseline.

### DS P0: Target and Objective Reformulation

- [ ] Add `log1p(sale_price)` target option.
- [ ] Add inverse-transform correction.
- [ ] Add `log_price_per_sqft` target option.
- [ ] Compare raw price, log price, and log PPSF targets.
- [ ] Optimize on MdAPE and PPE10, not only R2.
- [ ] Add a composite validation objective:
  - [ ] weighted segment MdAPE
  - [ ] PPE10
  - [ ] major slice floor
  - [ ] max slice regression
- [ ] Add rolling-origin validation.

### DS P1: Comparable Sales Engine

- [x] Build `src/features/comps.py` or equivalent.
- [x] Define comp eligibility:
  - [x] recent sale window
  - [x] same borough
  - [x] same/compatible property segment
  - [x] distance threshold
  - [x] sqft similarity
  - [x] age/year-built similarity
  - [x] unit-count similarity
  - [x] sale validity
- [x] Compute comp features:
  - [x] comp count
  - [x] median comp price
  - [x] median comp PPSF
  - [x] weighted comp estimate
  - [x] comp dispersion
  - [x] nearest comp distance
  - [x] median comp recency days
  - [x] local momentum
- [x] Persist selected comps for evaluation rows.
- [x] Expose top comps as model-package evidence.
- [ ] Add comp-count and comp-dispersion into confidence scoring.

### DS P1: Public Data Expansion

- [ ] Add PLUTO ingestion and BBL join.
- [ ] Add fields:
  - [ ] lot area
  - [ ] building area
  - [ ] number of floors
  - [ ] units
  - [ ] land use
  - [ ] zoning
  - [ ] FAR
  - [ ] year altered
  - [ ] historic district
  - [ ] lot frontage/depth
- [ ] Add DOB permit features:
  - [ ] recent permit count
  - [ ] major alteration flags
  - [ ] new construction flags
  - [ ] permit recency
- [ ] Add HPD/DOB/OATH violation features:
  - [ ] open violation count
  - [ ] severity-weighted count
  - [ ] recent violation count
  - [ ] resolved violation count
- [ ] Add geospatial features:
  - [ ] distance to subway
  - [ ] distance to park
  - [ ] flood zone flag
  - [ ] neighborhood geometry
- [ ] Add macro/time features:
  - [ ] mortgage-rate bucket
  - [ ] month/quarter
  - [ ] borough-level volume trend
  - [ ] local price index proxy

### DS P1: Confidence, Intervals, and Abstention

- [ ] Replace fixed +/-9% intervals.
- [ ] Add residual interval calibration by borough/segment/price band.
- [ ] Add conformal interval option.
- [ ] Track interval coverage:
  - [ ] 50% interval coverage
  - [ ] 80% interval coverage
  - [ ] 90% interval coverage
- [ ] Add confidence score model or rules.
- [ ] Confidence inputs should include:
  - [ ] comp count
  - [ ] comp recency
  - [ ] comp dispersion
  - [ ] feature completeness
  - [ ] OOD score
  - [ ] segment calibration
  - [ ] interval width
  - [ ] data freshness
- [ ] Add `hit_status`:
  - [ ] `hit`
  - [ ] `low_confidence_hit`
  - [ ] `no_hit`
- [ ] Add `abstention_reason`.
- [ ] Track hit rate by borough, segment, price band, and time.

### DS P1: AVM-Specific Metrics

- [x] Add PPE5.
- [x] Add PPE10.
- [x] Add PPE20.
- [x] Add MdAPE.
- [x] Add MAPE only as secondary due outlier sensitivity.
- [x] Add valuation ratio: predicted value / sale price.
- [x] Add coefficient of dispersion style metric.
- [x] Add PRD/PRB-style vertical equity diagnostics.
- [x] Add overvaluation rate.
- [x] Add undervaluation rate.
- [x] Add signed percentage error by slice.
- [x] Add interval coverage by slice when interval columns exist.
- [x] Add hit rate by slice when hit-status columns exist.

### DS P2: Fairness and Proxy Audit

- [ ] Define fairness audit scope and limitations.
- [ ] Use geography/value-band proxy analysis carefully.
- [ ] Evaluate valuation ratios by:
  - [ ] borough
  - [ ] neighborhood
  - [ ] census tract if available
  - [ ] value band
  - [ ] property segment
- [ ] Track systematic under/overvaluation.
- [ ] Track error gaps across slices.
- [ ] Document limitations in model card.
- [ ] Add promotion gates for unacceptable disparities.

### DS P2: Unsupervised and Deep Learning Showcase

- [ ] Add unsupervised property/submarket clusters.
- [ ] Add anomaly detection for likely non-market sales.
- [ ] Add OOD detector for unusual valuation requests.
- [ ] Add deep-learning only if suitable data is introduced:
  - [ ] listing text embeddings
  - [ ] image/condition scoring
  - [ ] neighborhood embeddings
- [ ] Keep deep-learning experiments separate from champion unless they pass governance gates.

## Frontend TODO

### Frontend P0: Model-Backed Valuation

- [x] Replace hard-coded borough PPSF valuation logic in the visible product experience with governed no-score behavior.
- [ ] Add backend scorer endpoint or service bridge.
- [ ] Return real model response:
  - [ ] valuation ID
  - [ ] model package ID
  - [ ] model alias
  - [ ] model version
  - [ ] route
  - [ ] feature contract version
  - [ ] predicted value
  - [ ] interval
  - [ ] confidence score
  - [ ] hit/no-hit status
  - [ ] abstention reason
  - [ ] evidence links
- [x] Keep heuristic mode only as explicit degraded fallback.
- [x] Add UI badge for `model-backed` vs `fallback`.

### Frontend P0: Artifact Transparency

- [x] Add model package detail page.
- [ ] Show:
  - [x] model card
  - [x] training manifest
  - [x] feature contract
  - [x] data manifest
  - [x] metrics
  - [x] slice scorecards
  - [x] artifact hashes
  - [x] release decision
- [x] Add artifact freshness warnings.
- [x] Add copyable package ID and run ID.

### Frontend P1: Explainability UX

- [ ] Replace hard-coded SHAP summary with generated SHAP JSON/CSV.
- [ ] Add local driver waterfall from actual inference explanation.
- [x] Add package-level comparable-sales evidence panel in the artifact explorer.
- [ ] Add top comparable sales table.
- [ ] Add comps on map.
- [ ] Add confidence decomposition:
  - [ ] comp coverage
  - [ ] feature completeness
  - [ ] OOD risk
  - [ ] interval width
  - [ ] segment calibration
- [ ] Add caveats near estimate output.
- [ ] Add "why no valuation" state for abstentions.

### Frontend P1: Governance UX

- [x] Add governed experiment workbench.
- [x] Show locked spec controls and immutable artifact paths.
- [x] Show previous experiment runs from `/api/v1/experiments/registry`.
- [x] Add review queue, experiment queue, and tracked experiment views.
- [x] Add action controls for review request, approval/rejection, training queue, and worker start.
- [x] Show lifecycle state in the active run output.
- [x] Add Playwright coverage for preflight, review, and training queue flows.
- [x] Add proposal summary/detail cards.
- [x] Show champion vs candidate comparison.
- [x] Show gate pass/fail evidence.
- [ ] Show AVM-specific gates:
  - [ ] hit rate
  - [ ] interval coverage
  - [ ] valuation ratio dispersion
  - [ ] fairness/proxy audit
  - [ ] random review status
- [x] Add approval/rejection form with local release-owner identity.
- [x] Require decision reason.
- [ ] Show audit log timeline.
- [ ] Show rollback target.

### Frontend P1: Monitoring UX

- [ ] Add prediction volume chart.
- [ ] Add hit-rate trend.
- [ ] Add confidence trend.
- [ ] Add interval coverage trend.
- [ ] Add performance by borough/segment/price band.
- [ ] Add drift root-cause table.
- [ ] Add stale data warning.
- [ ] Add retrain decision explanation.

### Frontend P2: Demo Polish

- [ ] Add a "model evidence drawer" available from valuation, governance, and monitoring.
- [ ] Add a guided reviewer flow:
  - [ ] open valuation
  - [ ] inspect comps
  - [ ] inspect drivers
  - [ ] inspect confidence
  - [ ] open model card
  - [ ] inspect governance gates
- [ ] Add a public-data limitation panel.
- [ ] Add not-for-lending/not-an-appraisal disclaimer.
- [ ] Add exportable valuation evidence report.

## Comprehensive Test Suite Plan

### Unit Tests

- [ ] Feature contract validation.
- [ ] Forbidden feature leakage.
- [ ] Point-in-time split behavior.
- [ ] Data manifest creation.
- [ ] Artifact hash generation.
- [ ] Model card required sections.
- [ ] Arena policy gate calculations.
- [ ] Audit log schema.
- [ ] Confidence score boundaries.
- [ ] Interval coverage calculation.
- [ ] Hit/no-hit policy.

### Integration Tests

- [ ] Train clean model package from small fixture.
- [ ] Validate full model package.
- [ ] Generate release decision.
- [ ] Register candidate.
- [ ] Generate proposal.
- [ ] Approve/reject proposal.
- [ ] Run scoring endpoint with model package.
- [ ] Write prediction log.
- [ ] Load frontend API contracts from artifact-backed payloads.

### End-to-End Tests

- [ ] Build data fixture.
- [ ] Train model.
- [ ] Validate package.
- [ ] Promote candidate to champion.
- [ ] Score a property.
- [ ] Produce explanation.
- [ ] Write prediction log.
- [ ] Show valuation in UI.
- [ ] Show governance evidence in UI.
- [ ] Show monitoring snapshot in UI.

## Immediate Next Steps

1. Build `docs/MODEL_ARTIFACT_CONTRACT.md`.
2. Build `src/mlops/artifact_contract.py`.
3. Add tests for feature leakage and artifact package validity.
4. Mark current artifacts as legacy.
5. Update release validation so only contract-valid packages can pass.
6. Retrain a clean `v2` baseline from scratch.
7. Generate a complete `v2` model package.
8. Wire the frontend valuation path to the real `v2` scorer.

The highest-leverage first implementation is the artifact contract validator. Once that exists, every DS and frontend improvement has a reliable evidence standard to target.
