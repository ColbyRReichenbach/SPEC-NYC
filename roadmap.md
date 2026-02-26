## Epic: LLM-EVID-001 — Grounded AVM Intelligence Layer (NYC-First)

**Status:** Planned  
**Owner:** ML + MLOps + AI Product  
**Decision Lock:** Phase 1 does **not** change numeric valuation; external signals are narrative/risk/context only.

### Goal
Add an auditable, citation-grounded LLM intelligence layer on top of the current AVM workflow, starting with NYC boroughs and scaling to multi-city via connector architecture.

### Product Promise
Auditable first-look intelligence:
1. Why this estimate?
2. What changed recently?
3. What should operators do next?
All answers must be grounded in approved evidence with explicit citations and freshness context.

### Milestones
| Milestone | Scope | Exit Criteria |
|---|---|---|
| M0 | Baseline hardening of current copilot + fallback + citation guard | Non-fallback responses always include citations; smoke validation green |
| M1 | Curated external evidence ingestion (daily schedule) | NYC allowlist sources ingested; evidence registry populated with source metadata + timestamps |
| M2 | Hybrid retrieval + typed citations + freshness status | Retrieval returns relevant docs with provenance; API includes `freshness_status` + structured citations |
| M3 | Event-trigger updates (permits/rates) | New high-signal updates ingested outside daily batch; alert events generated |
| M4 | Multi-city framework scaffolding | Connector interfaces and geo scoping generalized; NYC remains default production scope |
| M5 | Model-impact experimentation track (post-protocol) | Backtest + leakage protocol defined and passed before any signal touches price model |

### Backlog Hypotheses
1. `H-LLM-001` — External evidence ingestion framework (NYC allowlist)
2. `H-LLM-002` — Retrieval quality + strict citation enforcement
3. `H-LLM-003` — Event-trigger freshness pipeline (permits/rates)
4. `H-LLM-004` — Multi-city retrieval abstraction and geo-routing
5. `H-LLM-005` — External-signal model experiment (only after leakage-safe protocol)

### Non-Negotiables
1. No open-web free browsing for production responses.
2. Grounding-only responses from curated evidence + internal artifacts.
3. Missing/insufficient evidence must trigger deterministic fallback.
4. No direct price adjustment from external live signals in Phase 1.
5. All outputs remain auditable with source/date/provenance.

### Required API Additions (Planned)
1. Request context:
- `geo_scope` (`nyc` default)
- `include_external_signals` (`true` default)
- `freshness_preference` (`balanced` default)
2. Response fields:
- `external_evidence_used`
- `freshness_status` (`fresh|stale|partial`)
- `signal_summary`
- typed `citations[]` with `source_id`, `publisher`, `as_of_date`, `path_or_url`, `trust_tier`

### Evaluation and Gates
1. Retrieval relevance: intent-to-source match quality above threshold.
2. Safety: prompt-injection, out-of-scope, and citation-missing paths covered by tests.
3. Reliability: graceful degraded mode when external feeds are unavailable.
4. Ops: ingestion freshness SLA met (daily + event triggers).
5. Release: `python3 -m src.validate_release --mode smoke --contract-profile canonical` green after each integration increment.

### Risks and Mitigations
| Risk | Mitigation |
|---|---|
| Source noise/staleness | Source allowlist, trust tiers, freshness checks, stale caveats |
| Hallucinated claims | Citation-required invariant + fallback |
| Feed outage | Partial mode with internal artifact grounding |
| Leakage into model path | Policy lock: external signals narrative-only until backtested protocol |

### Review Cadence
1. Weekly roadmap review: milestone progress + blocked dependencies.
2. Biweekly experiment review: `H-LLM-*` status and evidence quality.
3. Promotion decision checkpoints only after gate pass evidence.

## ML Roadmap: Policy-Winning AVM Strategy (NYC-First, Scale-Ready)

### Summary
This roadmap focuses the next 8-12 weeks on one objective: produce a challenger that passes your arena policy gates, not just improves a headline metric.  
The strategy is a scalable two-track architecture: a strong global tabular baseline as production default, with controlled segmented/ensemble experiments only when slice coverage and pre-gates justify complexity.  
AI is used as a force multiplier for analysis, experiment ops, and diagnostics, while all model promotion decisions stay human-gated.

### Research-Aligned Positioning (What giants appear to do)
1. Large AVM players emphasize data quality, frequent refreshes, uncertainty/confidence, and operational risk controls over single model magic.
2. Public evidence suggests mixed modeling stacks and proprietary workflows, not one universal architecture everyone uses.
3. LLM/AI usage in real estate majors is strongest in product intelligence/search/support and workflow acceleration, while core valuation quality still depends on structured data + supervised modeling.
4. For your constraints (solo dev, open/public data, NYC first), the scalable choice is not copying one giant architecture, but building a governance-first system that can absorb richer data later without rewriting fundamentals.

### Primary Decisions Locked
1. Timeline: `8-12 weeks`.
2. Goal: `pass arena gates with challenger`.
3. Data spend: `open/public sources only`.
4. Architecture policy: `scale-ready two-track` (global default + controlled specialist experiments).
5. Tuning policy: `policy-aligned constrained Optuna`.
6. AI-in-DS policy: `copilot for analysis/ops, human-gated model decisions`.

### Direct Answers to Your 6 Questions
#### 1) Are we using the right metrics?
Current core metrics are directionally right (`PPE10`, `MdAPE`, segment/tier breakdown), and aligned with your policy gates.  
Gap: training/tuning objective is currently validation MdAPE only, while promotion uses weighted slice and floor/drop constraints.

Decision:
- Keep `PPE10` and `MdAPE` as primary policy metrics.
- Add secondary business-control metrics:
1. `WAPE` overall and by major segment.
2. `P90 APE` (tail error risk).
3. `coverage-weighted segment score` (volume-aware).
4. `confidence calibration` (error vs confidence band).
5. `stability deltas` over rolling quarters.

#### 2) Right model vs segmented/federated depth?
No single always-best model exists in AVM. For scale and maintainability, default to one strong tabular model plus guarded routing.

Decision:
- Production default: `global gradient-boosted tabular model`.
- Specialist routing only if pre-gates pass and sample thresholds are met.
- Do not drill to zip/neighborhood-specific models unless per-slice data sufficiency and stability thresholds are met over time.

#### 3) Multiple models/ensemble?
Yes, but staged.

Decision:
- Phase 1: single primary model + optional segmented router fallback.
- Phase 2 (only if justified): constrained ensemble (for example, weighted blend global + segment expert) with strict interpretability and rollback rules.
- No unconstrained many-model zoo in production.

#### 4) How to tune (Optuna etc.)?
Use Optuna, but objective must mirror business gates.

Decision:
- Replace single-objective tuning with constrained policy-aware objective:
- maximize `PPE10`
- minimize weighted segment `MdAPE`
- penalize `max_major_segment_ppe10_drop`
- penalize `major_segment_floor` breaches
- penalize stability regressions.
- Use time-aware CV (rolling/expanding windows), not random-only splits.

#### 5) More data sources vs more feature engineering?
Both, sequenced by ROI and leakage risk.

Decision:
- First: improve feature quality from existing/open NYC data.
- Next: add high-value open data sources (permits, transit access, local liquidity proxies, macro regime features).
- Keep strict inference-availability and leakage contracts for every feature.

#### 6) AI in data science workflows?
AI is becoming standard in DS workflows, but not a replacement for statistical rigor.

Decision:
- Use AI to accelerate:
1. experiment brief drafting,
2. failure forensics,
3. feature hypothesis generation,
4. report synthesis,
5. monitoring triage suggestions.
- Keep human ownership for target/feature definitions, experiment acceptance, and promotion decisions.
- Machine learning remains core; AI amplifies DS throughput.

### Implementation Roadmap (Decision-Complete)
#### Phase 0 (Week 1): Baseline Lock + Evaluation Contract Hardening
1. Freeze current champion baseline artifacts and acceptance baselines.
2. Create `evaluation_contract_v2` with required outputs:
- overall: `PPE10`, `MdAPE`, `R2`, `WAPE`, `P90_APE`
- per-segment/per-tier metrics
- rolling temporal stability metrics
- confidence calibration table.
3. Add pre-arena gate check command that fails fast before arena proposal if thresholds are violated.

#### Phase 1 (Weeks 2-3): Objective Alignment + Time-Aware Tuning
1. Implement constrained Optuna objective aligned to policy.
2. Add rolling time split evaluation for tuning and model selection.
3. Introduce major-slice guardrails in objective and trial pruning.
4. Emit trial-level gate diagnostics for each run.

#### Phase 2 (Weeks 3-5): Scalable Architecture Guardrails
1. Keep global model as primary.
2. Add routing eligibility contract:
- minimum train/test support per route,
- minimum stability over recent windows,
- automatic fallback to global when sparse/unstable.
3. Add architecture decision log to artifacts (`why global`, `why routed`, `why fallback`).

#### Phase 3 (Weeks 5-7): Feature/Data Expansion (Open Sources)
1. Prioritize features with high expected lift and low leakage risk:
- transit proximity,
- local liquidity/intensity,
- regime-safe temporal features,
- permit/change-intensity features (context first, model later if stable).
2. Add feature registry with mandatory metadata:
- inference availability,
- latency/freshness,
- leakage risk class,
- fallback behavior.
3. Enforce feature acceptance tests before entering training.

#### Phase 4 (Weeks 7-9): Controlled Ensemble Track (Optional)
1. Run ensemble experiments only if Phase 1-3 plateau.
2. Candidate ensemble types:
- blend global + routed expert,
- residual correction model on hard slices.
3. Require interpretability and rollback parity with current stack.

#### Phase 5 (Weeks 9-12): AI-Augmented DS Operations
1. Add AI copilot modules for:
- experiment recommendations from artifact history,
- auto-generated postmortems for failed challengers,
- monitoring remediation runbooks.
2. Enforce AI output is advisory.
3. Add telemetry for AI-assisted decisions and human override traces.

### Important Changes to Public Interfaces / Types / Artifacts
#### A) Metrics Schema (`models/metrics_*.json`)
Add:
1. `overall.wape`
2. `overall.p90_ape`
3. `stability.temporal_mdape_std`
4. `stability.temporal_ppe10_std`
5. `calibration.confidence_band_error`
6. `policy_precheck` object with pass/fail per gate.

#### B) Segment Scorecard (`reports/model/segment_scorecard_*.csv`)
Add columns:
1. `wape`
2. `p90_ape`
3. `coverage_weight`
4. `temporal_volatility_flag`.

#### C) New Pre-Arena Output
1. `reports/arena/pre_arena_gate_check_<tag>.json`
2. `reports/arena/pre_arena_gate_check_<tag>.md`
Purpose: block weak candidates before expensive governance workflow.

#### D) Training Metadata
Add to model metadata:
1. `selection_strategy` (`global|routed|ensemble`)
2. `route_eligibility_summary`
3. `objective_version`
4. `time_cv_scheme`.

### Test Cases and Scenarios
#### Metric/Policy Alignment
1. Tuning selects model that improves policy score, not just raw MdAPE.
2. Candidate with better MdAPE but floor breach is rejected in pre-arena check.

#### Architecture Guardrails
1. Sparse route automatically falls back to global.
2. Route with unstable temporal performance is disabled.

#### Feature Governance
1. Target-derived feature path is blocked.
2. Missing inference-time feature triggers fallback/validation error.
3. Feature registry without leakage metadata fails CI gate.

#### Temporal Robustness
1. Rolling-window validation detects regime instability.
2. Stability deltas reported and used in candidate acceptance.

#### Ensemble Safety
1. Ensemble candidate must beat baseline under identical gates.
2. Ensemble rollback path validated using saved fallback artifacts.

#### AI-in-DS Guardrails
1. AI recommendation without supporting citations is flagged advisory-only.
2. Human override decisions are logged with rationale.

### Acceptance Criteria (End of 8-12 Week Cycle)
1. At least one challenger passes all arena gates against current champion.
2. Pre-arena gate check blocks non-viable candidates reliably.
3. Time-aware evaluation and stability metrics are mandatory in every candidate package.
4. Routing/ensemble complexity is only enabled via explicit eligibility evidence.
5. Feature additions all pass leakage and inference-availability contracts.
6. AI assistance improves iteration speed without bypassing human governance.

### Explicit Assumptions and Defaults
1. Geography scope for model development remains NYC borough-focused in this cycle.
2. External paid data is out of scope; only open/public sources are used.
3. Core production model remains supervised tabular ML (GBDT family default).
4. No federated training architecture in this cycle; design remains compatible for future expansion.
5. Existing arena policy thresholds remain authoritative unless separately approved for change.
6. AI is a workflow multiplier, not an autonomous model promotion authority.

### External Reference Set Used for Planning
1. Zillow Zestimate/Neural Zestimate and data/update framing.
2. Redfin Estimate methodology/accuracy cadence.
3. Freddie Mac ACE/HVE collateral valuation governance framing.
4. ATTOM/HouseCanary AVM confidence and uncertainty positioning.
5. OpenAI file-search/prompt-caching and Anthropic contextual retrieval guidance for retrieval architecture trends.

## Arena Gate Calibration Roadmap (Policy v2 Proposal)

### Why This Is Needed
Current gate families are directionally correct and industry-aligned, but threshold calibration is mismatched to current dataset maturity and current champion evidence quality.

Observed in current artifacts:
1. Gate types are strong: weighted segment improvement, segment floor/drop, drift/fairness deltas.
2. Challenger failures are large (not near misses), so model/eval objective alignment is still the primary bottleneck.
3. Current champion artifact includes `price_tier` in `feature_columns`, which conflicts with current non-leakage direction and should be refreshed before final threshold locking.

### External Benchmark Context (for calibration, not direct copy)
1. Zillow and Redfin publish median error and within-X% metrics (business-interpretable and confidence-oriented).
2. Freddie emphasizes confidence tiers and out-of-sample validation for valuation trust.
3. OCC/interagency AVM rule emphasizes quality-control systems, random testing, anti-manipulation, and nondiscrimination controls rather than one universal threshold table.

### Policy v2 Design Principles
1. Keep gate families; recalibrate thresholds using your own baseline distribution.
2. Separate research progression gates from promotion gates.
3. Require confidence/stability metrics in addition to point-accuracy metrics.
4. Keep fairness/drift as no-regression constraints.

### Proposed `arena_policy_v2` (Concrete)

```yaml
registered_model_name: spec-nyc-avm
experiment_name: spec-nyc-avm
selection_window: 5
major_segment_min_n: 2000

# Stage-gate policy: research can progress with looser bounds,
# promotion remains strict and business-safe.
gates:
  research:
    weighted_segment_mdape_improvement: 0.00
    max_major_segment_ppe10_drop: 0.08
    major_segment_ppe10_floor: 0.20
    no_new_drift_alerts: true
    no_new_fairness_alerts: true
  promotion:
    weighted_segment_mdape_improvement: 0.01
    max_major_segment_ppe10_drop: 0.05
    major_segment_ppe10_floor: 0.22
    no_new_drift_alerts: true
    no_new_fairness_alerts: true

fairness:
  max_segment_ppe10_gap: 0.22

scoring:
  mdape_weight: 0.45
  ppe10_weight: 0.35
  stability_weight: 0.20

promotion:
  mode: semi_auto
  proposal_expiry_hours: 24
  approval_required: true
  auto_expire_pending: true

champion_quality:
  min_overall_ppe10: 0.24
  max_overall_mdape: 0.30

# New required reporting metrics for policy decisions
reporting_requirements:
  require_metrics:
    - overall.ppe10
    - overall.mdape
    - overall.wape
    - overall.p90_ape
    - stability.temporal_mdape_std
    - stability.temporal_ppe10_std
```

### Why These Values
1. `weighted_segment_mdape_improvement` from `0.05` to `0.01` for promotion:
- 5% weighted MdAPE uplift is currently too coarse for your observed model deltas.
- 1% keeps improvement pressure while avoiding impossible promotion in this maturity phase.
2. `max_major_segment_ppe10_drop` from `0.02` to `0.05`:
- 2-point drop max is very strict for heterogeneous NYC slices with current features.
- 5-point cap still blocks severe regressions.
3. `major_segment_ppe10_floor` from `0.24` to `0.22`:
- Better aligned with your current major segment distribution while still enforcing a quality floor.
4. `max_segment_ppe10_gap` from `0.20` to `0.22`:
- Small relaxation to avoid over-penalizing current segment heterogeneity, while keeping disparity control explicit.

### Hard Prerequisite Before Final Lock
1. Regenerate champion baseline with leakage-safe feature set and refreshed artifacts.
2. Recompute segment/tier/temporal distributions.
3. Validate that v2 thresholds are percentile-grounded (for example, anchored near p25-p40 historical challenger deltas for promotion strictness).

### Migration Plan
1. Create `config/arena_policy_v2.yaml` (do not overwrite v1 yet).
2. Add policy selector in arena workflow:
- `--arena-policy-path config/arena_policy_v2.yaml`
3. Run dual-track evaluation for 3-5 challenger cycles:
- record outcomes under v1 and v2.
4. If v2 shows better decision quality (fewer false negatives, no quality regressions), promote v2 to default.

### Validation Protocol for Gate Changes
1. Backtest past challengers under v2 and compare:
- accepted/rejected decisions,
- downstream quality risk,
- fairness/drift deltas.
2. Require no scenario where v2 would have approved a materially worse model on policy-critical slices.
3. Document gate-change rationale in `reports/arena/model_change_log.md`.

### Additional Metrics to Add Before v2 Goes Live
1. `overall.wape`
2. `overall.p90_ape`
3. `stability.temporal_mdape_std`
4. `stability.temporal_ppe10_std`
5. confidence calibration error by band

### Acceptance Criteria for Policy v2 Adoption
1. Leakage-safe champion refresh completed.
2. Dual-run comparison across >= 3 serious challengers completed.
3. No increase in approved models with unacceptable major-slice degradation.
4. Release validation remains green with v2 policy path.

### Industry References Used for Calibration Direction
1. Zillow Zestimate accuracy framing (median error + within-X%).
2. Redfin Estimate accuracy framing and update cadence.
3. Freddie confidence-level framing (confidence tiers + out-of-sample validation mindset).
4. OCC/interagency AVM quality-control standards (confidence, anti-manipulation, random testing, nondiscrimination).

## Execution Truth Audit and `.codex` Integration (ML + MLOps + Copilot)

### Current Truth Snapshot (Codebase)
1. ML model path is XGBoost tabular with time split, temporal features, segmented routing modes, and non-leaky `price_tier_proxy` enforcement.
2. Arena/gates already enforce weighted segment uplift, segment floor/drop, drift and fairness constraints.
3. Frontend copilot is evidence-grounded with required citations/fallback, safety/audit/telemetry/session memory.
4. Copilot context currently references mostly static internal artifacts and does not yet ingest live external market sources.
5. Champion metrics artifact currently lists `price_tier` in feature columns and should be refreshed from current leakage-safe training path before final policy lock.

### Gaps Not Yet Implemented
1. Policy-aligned tuning objective (current Optuna objective is validation MdAPE only).
2. Pre-arena fail-fast artifact (`pre_arena_gate_check`) before proposal generation.
3. Two-level gate strategy (`research` vs `promotion`) in active policy.
4. Expanded business-risk metrics (`WAPE`, `P90_APE`, confidence calibration).

### De-duplication Rule
1. `roadmap.md` is the single strategy and execution source.
2. Keep implementation details here concise and linked to workflows; avoid duplicating long narrative sections.

### `.codex` Mapping (Agents/Workflows)
1. Agents:
- `.codex/agents/ml_engineer.md`
- `.codex/agents/data_engineer.md`
- `.codex/agents/qa_release.md`
- `.codex/agents/orchestrator.md`
2. Workflows:
- `.codex/workflows/20_feature_iteration.md`
- `.codex/workflows/30_model_iteration.md`
- `.codex/workflows/50_release_candidate.md`
- `.codex/workflows/60_autopilot_loop.md`

### Self-Healing Red-to-Green Loop
1. Run targeted tests for changed scope.
2. End each iteration with:
- `python3 -m src.validate_release --mode smoke --contract-profile canonical`
3. If red:
- classify failure (`Data | Model | Product | Ops`)
- apply focused fix
- rerun failed checks + smoke validation
4. Repeat until green or iteration cap.
5. Optional automation:
- `./scripts/autonomy_loop.sh --max-iterations 3`
- optional pack: `config/autonomy/repair_pack.example.json`

### Definition of Done per Model Iteration
1. Champion/challenger comparison with overall + segment + tier metrics.
2. Machine-readable gate decision output.
3. SHAP artifacts linked in evidence.
4. Drift/missingness diagnostics generated.
5. Smoke release validation green at iteration end.
6. Decision memo recorded (`promote | hold | reject | rollback`).

### Immediate Next Actions
1. Refresh champion with leakage-safe feature contract and re-emit artifacts.
2. Implement policy-aware tuning objective.
3. Add and enforce pre-arena gate-check artifact.
4. Expand metrics (`WAPE`, `P90_APE`, temporal stability, calibration).
5. Re-run arena with refreshed baseline and reassess thresholds from fresh evidence.

## Codex-Executable Development Plan (Fully Mapped)

### Phase Matrix (Owner + Workflow + Checklist + Gates)
1. Phase `P0 Bootstrap`
- Owner agent: `.codex/agents/orchestrator.md`
- Workflow: `.codex/workflows/00_bootstrap.md`
- Checklist: `.codex/checklists/pr_template.md`
- Required gates:
  - `python3 -m compileall src web`
  - `python3 -m src.validate_release --mode smoke --contract-profile canonical`
2. Phase `P1 Datasource and Contract Hardening`
- Owner agents: `.codex/agents/data_engineer.md`, `.codex/agents/qa_release.md`
- Workflow: `.codex/workflows/10_datasource_onboarding.md`
- Checklist: `.codex/checklists/datasource_intake_checklist.md`
- Required gates:
  - canonicalization + contract checks
  - `python3 -m src.validate_release --mode smoke --contract-profile canonical`
3. Phase `P2 Feature and Leakage Controls`
- Owner agents: `.codex/agents/data_engineer.md`, `.codex/agents/ml_engineer.md`
- Workflow: `.codex/workflows/20_feature_iteration.md`
- Checklist: `.codex/checklists/model_change_checklist.md`
- Required gates:
  - leakage tests + inference-availability checks
  - drift/missingness diagnostics by segment/time
  - `python3 -m src.validate_release --mode smoke --contract-profile canonical`
4. Phase `P3 Model Iteration and Arena Evidence`
- Owner agents: `.codex/agents/ml_engineer.md`, `.codex/agents/orchestrator.md`
- Workflow: `.codex/workflows/30_model_iteration.md`
- Checklist: `.codex/checklists/model_change_checklist.md`
- Required gates:
  - challenger vs champion metrics (overall/segment/tier/stability)
  - arena proposal artifacts with dataset/feature version tags
  - `python3 -m src.validate_release --mode smoke --contract-profile canonical`
5. Phase `P4 Frontend and Product Evidence Surface`
- Owner agents: `.codex/agents/frontend_engineer.md`, `.codex/agents/backend_engineer.md`
- Workflow: `.codex/workflows/40_ui_iteration.md`
- Checklist: `.codex/checklists/pr_template.md`
- Required gates:
  - `npm run -w web lint`
  - `npm run -w web typecheck`
  - `npm run -w web build`
  - `python3 -m src.validate_release --mode smoke --contract-profile canonical`
6. Phase `P5 Release Candidate and Promotion Decision`
- Owner agents: `.codex/agents/qa_release.md`, `.codex/agents/orchestrator.md`
- Workflow: `.codex/workflows/50_release_candidate.md`
- Checklist: `.codex/checklists/model_change_checklist.md`
- Required gates:
  - gate-by-gate policy decision (`approve/reject`)
  - production readiness checks
  - `python3 -m src.validate_release --mode smoke --contract-profile canonical`
7. Phase `P6 Autopilot Remediation Loop`
- Owner agent: `.codex/agents/orchestrator.md`
- Workflow: `.codex/workflows/60_autopilot_loop.md`
- Checklist: `.codex/CODEX.md` invariant
- Required gates:
  - `./scripts/autonomy_loop.sh --max-iterations 3`
  - each attempt ends with `python3 -m src.validate_release --mode smoke --contract-profile canonical`

### Anti-Drift Enforcement Rules
1. Every roadmap item must reference one primary `.codex/workflows/*` file before implementation starts.
2. Every merged change set must reference one checklist file under `.codex/checklists/*`.
3. Any failed gate is classified as `Data | Model | Product | Ops` and logged with remediation command(s).
4. No promotion claim is valid without machine-readable gate outputs and artifact paths.

### Operator Runbook (Minimal)
1. Bootstrap: `python3 -m compileall src web`
2. Frontend CI gates: `npm run -w web lint && npm run -w web typecheck && npm run -w web build`
3. Release smoke gate: `python3 -m src.validate_release --mode smoke --contract-profile canonical`
4. Optional autonomous retries: `./scripts/autonomy_loop.sh --max-iterations 3`
