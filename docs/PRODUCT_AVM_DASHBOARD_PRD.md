# Production AVM Dashboard PRD (Non-Streamlit)

Status: Draft v1  
Owner: Product + Frontend Lead  
Date: 2026-02-23  
Scope: Production web app for valuation, governance, and monitoring
Related specs:
- `docs/FRONTEND_ARCHITECTURE_RFC.md`
- `docs/AI_EXPLAINABILITY_COPILOT_SPEC.md`

## 1) Product Intent

Build a production-grade AVM dashboard that replaces Streamlit with a modern web application.  
The product must support valuation workflows and model governance in one operational surface.

Design direction:
- Premium, editorial-fintech visual language (Azuli-inspired quality bar)
- Dense information hierarchy without feeling cluttered
- Fast, confident interactions for operator workflows
- Auditable evidence links on every model decision surface

## 2) Users and Jobs To Be Done

Primary users:
- Analyst: run single and batch valuations with confidence context
- MLOps/Release owner: evaluate challenger status and promotion readiness
- Risk/Governance reviewer: inspect evidence, gates, and monitoring health

Core jobs:
1. Price one property with contextual confidence and comps-like signals.
2. Submit large property sets for batch scoring and download results.
3. Decide promote/reject using policy gates and linked evidence.
4. Detect drift and performance degradation early by segment and time.

## 3) Core Pages

### A) Single Valuation
Purpose:
- Fast, high-confidence single-property valuation.

Key modules:
- Property input form (address + structured fields)
- Valuation output card (predicted value, interval, PPE band hint)
- Context panel (segment, tier proxy, route/model used, key drivers)
- Explainability card (top positive/negative SHAP drivers)
- AI copilot panel ("why this estimate?", confidence improvement guidance)
- Evidence links (run card, metrics JSON, SHAP artifact)

Primary actions:
- Run valuation
- Save valuation record
- Export one-page valuation report (PDF)

### B) Batch Valuation
Purpose:
- Upload and score many properties with job visibility.

Key modules:
- File upload + schema validator
- Batch job queue table (submitted, running, done, failed)
- Result summary (count, success rate, latency, error reasons)
- Download center (predictions CSV, QA report, failure rows)

Primary actions:
- Start batch job
- Retry failed rows
- Download outputs

### C) Model Governance
Purpose:
- Champion/challenger lifecycle management in one place.

Key modules:
- Registry alias panel (champion, challenger, candidate)
- Policy gate board (pass/fail by gate with thresholds)
- Proposal timeline (pending, approved, rejected, expired, no_winner)
- Promotion decision panel (approve/reject with reason capture)
- Copilot governance assistant ("what changed recently?"; gate interpretation)

Primary actions:
- Generate proposal
- Review gate diffs
- Approve or reject (with audit note)

### D) Monitoring and Drift
Purpose:
- Operational health and model stability tracking.

Key modules:
- Drift summary (global + feature-level alerts)
- Performance trends (PPE10, MdAPE, R2 by time)
- Slice diagnostics (segment, tier, borough)
- Retrain recommendation card (from policy artifact)
- Copilot monitoring assistant (root-cause hints + remediation suggestions)

Primary actions:
- Filter timeframe/slice
- Drill into alert root cause
- Open remediation runbook

## 4) Layout Hierarchy and Flows

## Navigation IA
- Left rail: `Valuation`, `Batch`, `Governance`, `Monitoring`, `Artifacts`
- Top bar: environment badge, model alias, dataset version, user, quick search
- Global right drawer: artifact inspector (opens JSON/MD/CSV references)

## Page Hierarchy
1. Header context strip (model version, dataset version, last updated)
2. Primary action zone (run valuation, start batch, approve proposal)
3. KPI strip (headline metrics and gate status)
4. Diagnostic workspace (charts, tables, explanations)
5. Audit trail footer (who/when/action/artifact links)

## User Flows

Flow 1: Single valuation
1. Enter address + required fields.
2. Client-side validation + server-side validation.
3. Receive valuation with confidence context and model route.
4. Save and export valuation report.

Flow 2: Batch valuation
1. Upload file and map columns.
2. Resolve schema issues.
3. Submit async job and track progress.
4. Download scored output and failure report.

Flow 3: Governance decision
1. Open latest proposal from Governance page.
2. Review gate board and segment/tier deltas.
3. Open run evidence from artifact drawer.
4. Approve or reject with reason.

Flow 4: Monitoring response
1. Alert appears in Monitoring summary.
2. Drill into affected segment/time window.
3. Confirm impact level and remediation path.
4. Trigger follow-up workflow ticket.

## 5) Design Token System

## Color tokens
```yaml
color:
  bg:
    canvas: "#F6F8F7"
    surface: "#FFFFFF"
    elevated: "#EEF2EF"
  text:
    primary: "#111A16"
    secondary: "#43524B"
    muted: "#6B7B73"
    inverse: "#F3F6F4"
  brand:
    primary: "#0E8A6A"
    primary_hover: "#0B7459"
    accent: "#D9A441"
  status:
    success: "#1E9E63"
    warning: "#C7821F"
    danger: "#C7463D"
    info: "#2C7FB8"
  border:
    subtle: "#D8E0DB"
    strong: "#AAB8B0"
```

## Typography tokens
```yaml
typography:
  family:
    display: "\"Space Grotesk\", \"Sora\", sans-serif"
    body: "\"Manrope\", \"IBM Plex Sans\", sans-serif"
    mono: "\"IBM Plex Mono\", monospace"
  size:
    xs: 12
    sm: 14
    md: 16
    lg: 20
    xl: 28
    xxl: 40
  weight:
    regular: 400
    medium: 500
    semibold: 600
    bold: 700
  line_height:
    tight: 1.2
    normal: 1.45
    relaxed: 1.6
```

## Spacing, radius, shadow, motion
```yaml
spacing:
  2: 8
  3: 12
  4: 16
  5: 20
  6: 24
  8: 32
  10: 40
  12: 48
radius:
  sm: 8
  md: 12
  lg: 16
  pill: 999
shadow:
  card: "0 6px 24px rgba(17, 26, 22, 0.08)"
  modal: "0 12px 40px rgba(17, 26, 22, 0.16)"
motion:
  fast: "120ms ease-out"
  standard: "220ms cubic-bezier(0.2, 0.8, 0.2, 1)"
```

## Interaction principles
- Use staged reveal (not noisy micro-animations)
- Prefer skeletons over spinners for data-heavy panels
- Keep gate state color + icon + text (never color-only)
- Maintain keyboard and screen reader accessibility
- Show explainability caveats and confidence caveats near valuation output

## 6) Backend API Contracts

Base path: `/api/v1`  
Auth: JWT bearer token  
All write endpoints return `request_id` for audit tracing.

### A) Single valuation
- `POST /valuations/single`
Request:
```json
{
  "property": {
    "address": "123 Example St, Brooklyn, NY",
    "borough": "BROOKLYN",
    "gross_square_feet": 1800,
    "year_built": 1930,
    "residential_units": 2,
    "total_units": 2,
    "building_class": "B1",
    "property_segment": "SMALL_MULTI",
    "sale_date": "2026-02-23"
  },
  "context": {
    "dataset_version": "ds_hseason001_train_20260223",
    "model_alias": "champion"
  }
}
```
Response:
```json
{
  "valuation_id": "val_01H...",
  "predicted_price": 1285000,
  "prediction_interval": {"low": 1170000, "high": 1399000},
  "confidence_band": "medium",
  "model": {
    "alias": "champion",
    "run_id": "34e917e198af4e58adb2097b8d9ca229",
    "model_version": "1",
    "route": "SMALL_MULTI"
  },
  "evidence": {
    "run_card_path": "reports/arena/run_card_34e9....md",
    "metrics_path": "models/metrics_v1.json",
    "shap_summary_path": "reports/model/shap_summary_v1.png"
  }
}
```

### B) Batch valuation
- `POST /valuations/batch` (multipart upload)
- `GET /valuations/batch/{job_id}`
- `GET /valuations/batch/{job_id}/results`
- `GET /valuations/batch/{job_id}/errors`

Batch status response:
```json
{
  "job_id": "job_01H...",
  "status": "running",
  "submitted_at": "2026-02-23T16:10:00Z",
  "processed_rows": 4200,
  "total_rows": 12000,
  "success_rows": 4125,
  "error_rows": 75
}
```

### C) Governance
- `GET /governance/status`
- `POST /governance/proposals/generate`
- `GET /governance/proposals/latest`
- `POST /governance/proposals/{proposal_id}/approve`
- `POST /governance/proposals/{proposal_id}/reject`

Latest proposal response:
```json
{
  "proposal_id": "57e6c66f5205",
  "status": "no_winner",
  "champion": {"run_id": "34e9...", "model_version": "1"},
  "winner": null,
  "candidates_ranked": [
    {
      "run_id": "879ab7838c214d3a907e34a687978264",
      "gate_pass": false,
      "weighted_segment_mdape_improvement": -0.7001,
      "overall_ppe10_lift": -0.1068,
      "max_major_segment_ppe10_drop": 0.1268,
      "min_major_segment_ppe10": 0.1219
    }
  ]
}
```

### D) Monitoring
- `GET /monitoring/overview?window=30d`
- `GET /monitoring/drift?window=30d&segment=ELEVATOR`
- `GET /monitoring/performance?window=30d&slice=segment`
- `GET /monitoring/retrain-decision/latest`

### E) Explainability + Copilot
- `POST /explanations/property`
- `POST /copilot/ask`

## 7) Sprint Plan and Acceptance Criteria

## Sprint 1: Foundation and shell (2 weeks)
Deliverables:
- Frontend app shell (Next.js + TypeScript + component system)
- Auth, routing, layout scaffolding, design tokens in code
- Artifact drawer and global context strip

Acceptance criteria:
1. All four core page routes exist behind auth.
2. Token system applied to primary UI components.
3. Lighthouse performance score >= 85 on desktop for shell pages.

## Sprint 2: Valuation workflows (2 weeks)
Deliverables:
- Single valuation page end-to-end
- Batch upload, validation, and job status views
- CSV download + error breakdown
- Property-level explanation card + confidence/caveat display

Acceptance criteria:
1. Single valuation request round-trip under 2.5s p95.
2. Batch job state updates within 5s polling interval.
3. Validation errors surfaced with row-level detail.

## Sprint 3: Governance and monitoring (2 weeks)
Deliverables:
- Governance gate board and proposal timeline
- Monitoring dashboards with drift/performance slices
- Deep links to artifacts and run cards
- Copilot integration on valuation/governance/monitoring views

Acceptance criteria:
1. Gate table exactly matches policy thresholds from backend.
2. Approval/rejection actions create auditable reason record.
3. Monitoring charts support time + segment filters.

## Sprint 4: Hardening and release (2 weeks)
Deliverables:
- Accessibility pass (WCAG AA target)
- Error boundaries, observability, and SLO dashboards
- E2E regression suite and release runbook

Acceptance criteria:
1. Critical user flows covered by E2E tests (single, batch, approve/reject).
2. No P0/P1 accessibility violations in audit.
3. Production release checklist complete with rollback plan.

## 8) Non-Functional Requirements

- Security: JWT auth, RBAC for governance actions, immutable audit logs
- Reliability: 99.9% API availability for valuation endpoints
- Performance: <300ms p95 for governance reads, <2.5s p95 single valuation
- Observability: request traces + frontend error telemetry + synthetic checks
- Compliance: evidence path persistence for all governance decisions

## 9) Risks and Mitigations

1. Risk: API contract drift from MLOps artifacts  
Mitigation: OpenAPI schema tests in CI + typed frontend SDK generation.

2. Risk: Governance action misuse  
Mitigation: RBAC, dual-confirm dialogs, mandatory reason on decision actions.

3. Risk: Monitoring trust gap due stale data  
Mitigation: freshness badges and hard stale-state warnings.

## 10) Out of Scope (v1)

- Manual model editing from UI
- Cross-city multi-tenant support
- Full PDF report designer
