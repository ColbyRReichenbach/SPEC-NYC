# Fully Dynamic AVM Platform TODO

This checklist converts the static/demo audit into implementation work. A production-grade demo must use one backend contract for model package selection, scoring, explanations, governance, monitoring, and UI state. No page should claim work is wired if the backend is returning fixed demo values.

## P0 - Contract Boundaries

- [x] Add a canonical model-package resolver for champion/candidate/challenger selection.
- [x] Stop dashboard package loading from picking the latest directory blindly.
- [x] Make visible package state describe its actual resolver source and fallback reason.
- [x] Move platform option lists out of React components and into backend-readable config.
- [ ] Add authenticated actor identity. Current implementation uses local config identity because this repo has no auth provider.

## P0 - Real Valuation And Explanation

- [x] Replace TypeScript borough-PPSF valuation heuristic with a model-backed Python scoring boundary.
- [x] Score against the selected model package's `model.joblib`, `feature_contract.json`, `metrics.json`, and `data_manifest.json`.
- [x] Generate feature vectors from inference-safe fields and strict as-of training-source artifacts.
- [x] Return model-backed local contribution drivers from XGBoost contribution values where available.
- [x] Persist valuation request/response artifacts under `reports/valuations`.
- [x] Make the property explanation endpoint read persisted valuation artifacts instead of returning fixed drivers.
- [ ] Add rich user-facing entry for optional geocode/H3 inputs. Current request contract still accepts the original minimal public-record fields.

## P0 - UI Dynamism

- [x] Wire the valuation workbench to `/api/v1/valuations/single`.
- [x] Display actual prediction, interval, route, confidence, evidence, and drivers.
- [x] Replace hardcoded evidence-state heading/status with package decision and resolver state.
- [x] Make experiment form options/presets derive from backend data/config rather than local arrays only.
- [x] Make release decisions use configured local actor identity rather than hardcoded component literals.
- [x] Replace static Batch/Copilot placeholder pages with dynamic operational state.

## P1 - Governance And Monitoring Unification

- [x] Replace legacy governance status adapter reading `reports/arena/proposal_*.json` with the current governance registry.
- [x] Return `null` for governance metrics that are not present in the current release artifacts instead of emitting fake zero deltas.
- [x] Make global explainability endpoint read current package model importance/contribution artifacts instead of static feature lists.
- [x] Point property catalog loading at the selected package data source before legacy dataset fallback.
- [ ] Persist expired proposal state when auto-expiring. Current registry reports expiration dynamically but does not rewrite the proposal file.
- [ ] Add duplicate release-proposal prevention for the same completed experiment/package pair.

## P1 - Batch Workflows

- [x] Replace fake batch job responses with filesystem-backed batch job artifacts.
- [x] Process submitted valuation requests through the same model-backed scoring path.
- [x] Read batch job status from persisted artifacts.
- [ ] Add CSV upload UI. Current batch contract is JSON API-first.

## P1 - Artifact Integrity

- [ ] Verify artifact viewer contents against package `artifact_hashes.json`.
- [ ] Surface artifact parse errors instead of silently degrading to empty tables.
- [ ] Add package/history selectors for EDA, packages, proposals, and experiments.

## P2 - Production Hardening

- [ ] Replace local filesystem registries with durable database-backed registries for concurrent users.
- [ ] Add authz checks for review approval, release approval, and worker start.
- [ ] Add async queue infrastructure instead of detached local subprocess workers.
- [ ] Add full notebook rendering for rich outputs/images.
