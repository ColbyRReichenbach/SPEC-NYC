# Codex Agentic Setup for Production AVM Delivery (Azuli.ai Internship Track)

## Purpose
This guide adapts SPEC-NYC into a Codex-first, low-human-interaction engineering system that can ingest unknown schemas, iterate safely, and ship a production-grade AVM with auditable decisions.

It is designed for:
- rapid onboarding to a new dataset from Azuli.ai,
- autonomous implementation loops (plan → build → evaluate → propose),
- explicit human gates only at critical business/risk checkpoints.

---

## 1) Codex Operating Model (Agentic but Controlled)

### Core principle
Use **bounded autonomy**:
- agents can plan/implement/test/document on their own,
- promotion/deployment and policy exceptions require human approval.

### Recommended role split
1. **Orchestrator agent**
   - Owns task decomposition, branch strategy, and acceptance criteria.
   - Emits machine-readable task plans.
2. **Data agent**
   - Builds adapters for new source systems and data contracts.
   - Tracks schema drift and feature availability.
3. **Modeling agent**
   - Runs feature experiments, hyperparameter search, and segmentation strategy changes.
   - Produces performance deltas + explainability artifacts.
4. **MLOps/Release agent**
   - Enforces arena policy, retrain policy, release readiness checks.
   - Generates run cards, release notes, and rollback specs.
5. **Product/UI agent**
   - Maintains frontend behavior and Azuli branding consistency.

Use one orchestrator and multiple specialist agents as handoff targets.

---

## 2) Repo Conventions to Enable Codex Autonomy

### A. Add stable machine-readable contracts
Keep these as first-class artifacts:
- `contracts/data/*.yaml` for source schemas and validation thresholds.
- `contracts/features/*.yaml` for feature definitions and owner.
- `contracts/evaluation/*.yaml` for pass/fail gates (global + segment metrics).
- `contracts/release/*.yaml` for deployment policy.

### B. Enforce deterministic task units
Each autonomous task should include:
- objective,
- touched files,
- acceptance tests,
- rollback plan,
- evidence paths.

Store in `plans/tasks/*.md`.

### C. Structured run ledger
Write every run to `reports/runs/<timestamp>_<id>/`:
- config snapshot,
- data/profile summary,
- metrics,
- error analysis,
- recommendation (`promote`/`hold`/`rollback`).

This allows another agent to continue work without re-discovery.

---

## 3) Unknown Schema Adaptation Strategy (Critical for Internship Data)

When Azuli provides unknown tables/files/APIs:

1. **Source profiling stage (no training yet)**
   - infer types, null rates, cardinality, timestamp coverage,
   - detect likely target leakage fields (sale price proxies, post-close fields).
2. **Schema mapper stage**
   - map raw fields to canonical AVM entities:
     - property,
     - transaction,
     - listing,
     - neighborhood,
     - macro context.
3. **Contract generation stage**
   - generate initial data contracts automatically,
   - require human sign-off only once for baseline contract.
4. **Feature viability stage**
   - classify features as:
     - train-time safe,
     - inference-safe,
     - restricted (leaky/late-arriving).
5. **Training eligibility gate**
   - block runs until minimum coverage thresholds are met.

This minimizes fragile one-off scripts and allows reusable ingestion for new data sources.

---

## 4) Autonomous Iteration Loop (Codex-friendly)

### Loop template
1. Plan experiment from backlog hypothesis.
2. Implement minimal diff.
3. Run static checks + tests + evaluation.
4. Compare challenger vs champion.
5. Produce decision memo and next action.

### Stopping criteria
Autonomous loop should pause for human decision only when:
- policy threshold crossed,
- fairness/risk alert triggered,
- major data contract breakage,
- production deployment requested.

---

## 5) OpenAI/Codex Best-Practice Alignment

This setup aligns with mainstream OpenAI platform guidance patterns:
- keep prompts and instructions explicit, scoped, and versioned,
- use tool-calling for deterministic actions (data pulls, checks, report generation),
- preserve traceability (inputs, outputs, decisions),
- separate planning from execution,
- enforce eval-driven development before release.

Implementation interpretation in this repo:
- policy as files (`config/`, `docs/`, future `contracts/`),
- artifacts as evidence (`reports/`),
- test and release gates as safety boundaries (`tests/`, `src/validate_release.py`).

---

## 6) Frontend + Branding Track for Azuli.ai

Because branding can shift, implement a **theme token layer** instead of hardcoded styling:

- `frontend/theme/tokens.json`
  - colors (primary, secondary, accent, neutral)
  - typography families/sizes
  - spacing/radius/shadows
- load tokens into Streamlit or chosen frontend framework.

### Brand ingestion workflow
1. Capture current brand tokens from Azuli design references/site.
2. Save versioned token file (`v1`, `v2`, ...).
3. Generate UI from tokens only.
4. Snapshot screenshot artifacts for review.

Note: network scraping was blocked in this environment, so exact brand hex/type extraction could not be completed here.

---

## 7) Recommended Near-Term Deliverables

1. **Canonical schema layer**
   - add data model docs for `property`, `sale_event`, `geo_context`, `listing_context`.
2. **Adapter interface**
   - create pluggable source adapters for CSV/API/DB.
3. **Evaluation contract**
   - codify minimum production thresholds by market segment.
4. **Model registry hardening**
   - require signed run cards before alias updates.
5. **UI hardening**
   - production-grade prediction endpoint + input validation + confidence/explanation display.

---

## 8) Should You Start a New Repo?

**Recommendation: Keep SPEC-NYC as base and create a focused branch (or mono-repo subpackage) for Azuli internship delivery.**

Why:
- SPEC-NYC already contains useful MLOps patterns (arena, retrain policy, release validation).
- Starting over increases delivery risk and reduces visible execution depth.
- A targeted “Azuli adapter layer” demonstrates real-world integration skill, which is stronger for internship signaling.

Create a new repo only if:
- legal/data isolation is required,
- company policy forbids external history,
- architecture must diverge substantially (e.g., strict microservice deployment constraints).

