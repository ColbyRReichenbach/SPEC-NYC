# Frontend Architecture RFC

Status: Proposed  
Owner: Product + Frontend Lead  
Date: 2026-02-23  
Related PRD: `docs/PRODUCT_AVM_DASHBOARD_PRD.md`
Related AI spec: `docs/AI_EXPLAINABILITY_COPILOT_SPEC.md`

## 1) Decision Summary

Adopt a production web stack with:
- Next.js (App Router, TypeScript) for UI shell and routing
- BFF layer inside Next.js route handlers for frontend-safe API orchestration
- Shared design-system package for reusable tokens and components
- Query/cache layer via TanStack Query for robust data-fetch UX

This replaces Streamlit for stakeholder and operator-facing surfaces.

## 2) Goals and Non-Goals

Goals:
1. Production-grade UX for valuation, governance, and monitoring.
2. Strong separation of presentation, orchestration, and domain contracts.
3. Stable component system with consistent token usage.
4. Auditable flows that preserve artifact links and governance context.

Non-goals:
1. Rewriting model training or arena backend logic.
2. Building a generic CMS/report-builder in v1.
3. Multi-tenant architecture in first release.

## 3) High-Level Architecture

Client (Browser)
-> Next.js App Router UI  
-> Next.js BFF route handlers (`/api/*`)  
-> Existing backend services (valuation, governance, monitoring artifacts)

Reasoning:
- Keeps backend credentials and internal paths out of browser.
- Enables response normalization and contract versioning.
- Centralizes auth/session handling and audit metadata injection.

## 4) Runtime and Core Tech

- Framework: Next.js 15+ (App Router)
- Language: TypeScript (strict mode)
- Styling: Tailwind + CSS variables from design tokens
- Data fetching: TanStack Query + server actions where useful
- Charts: Recharts or ECharts (time/slice diagnostics)
- Tables: TanStack Table
- Forms: React Hook Form + Zod
- Testing: Vitest + React Testing Library + Playwright
- Telemetry: OpenTelemetry browser/server hooks + Sentry

## 5) Repo Layout (Proposed)

```text
web/
  app/
    (marketing)/
    (dashboard)/
      valuation/
        single/page.tsx
        batch/page.tsx
      governance/
        page.tsx
      monitoring/
        page.tsx
      copilot/
        page.tsx
      artifacts/
        page.tsx
      layout.tsx
    api/
      valuations/
        single/route.ts
        batch/route.ts
        batch/[jobId]/route.ts
      governance/
        status/route.ts
        proposals/generate/route.ts
        proposals/latest/route.ts
        proposals/[proposalId]/approve/route.ts
        proposals/[proposalId]/reject/route.ts
      monitoring/
        overview/route.ts
        drift/route.ts
        performance/route.ts
        retrain-decision/latest/route.ts
      explanations/
        property/route.ts
      copilot/
        ask/route.ts
    globals.css
    layout.tsx

  src/
    bff/
      clients/
        valuationClient.ts
        governanceClient.ts
        monitoringClient.ts
      mappers/
        valuationMapper.ts
        governanceMapper.ts
        monitoringMapper.ts
      errors/
        apiError.ts
      auth/
        session.ts
      audit/
        requestContext.ts
    features/
      valuation/
        components/
        hooks/
        schemas/
        services/
      batch/
        components/
        hooks/
        schemas/
        services/
      governance/
        components/
        hooks/
        schemas/
        services/
      monitoring/
        components/
        hooks/
        schemas/
        services/
      copilot/
        components/
        hooks/
        schemas/
        services/
    components/
      layout/
      navigation/
      charts/
      tables/
      feedback/
      artifacts/
    lib/
      queryClient.ts
      env.ts
      formatters.ts
    types/
      api.ts
      domain.ts

packages/
  design-system/
    src/
      tokens/
        colors.ts
        typography.ts
        spacing.ts
        radii.ts
        shadows.ts
      primitives/
        Button.tsx
        Input.tsx
        Select.tsx
        Badge.tsx
        Card.tsx
        Modal.tsx
        Tabs.tsx
        Tooltip.tsx
      patterns/
        KPIStat.tsx
        GateStatusPill.tsx
        PageHeader.tsx
        DataTableShell.tsx
      index.ts
    package.json
```

## 6) BFF Contract Strategy

BFF responsibilities:
1. Authenticate/authorize request.
2. Call backend service(s).
3. Normalize and redact payload fields for frontend.
4. Attach `request_id`, timestamp, and model context metadata.
5. Return typed DTOs with stable versions.

Versioning approach:
- Route contracts namespaced by `/api/v1`.
- Add `x-contract-version` header on responses.
- Use Zod schemas at boundary to reject malformed upstream data.

## 7) Component Map by Page

## Single Valuation
- `PageHeader`
- `PropertyInputForm`
- `ValuationResultCard`
- `ConfidenceBandCard`
- `ModelRouteCard`
- `KeyDriversCard`
- `SHAPDriversCard`
- `CopilotPanel`
- `ArtifactLinksPanel`

## Batch Valuation
- `BatchUploadDropzone`
- `SchemaMappingAssistant`
- `BatchJobTable`
- `BatchProgressCard`
- `BatchResultSummary`
- `BatchErrorTable`

## Governance
- `AliasStatusPanel`
- `ProposalTimeline`
- `GateBoardTable`
- `CandidateComparisonTable`
- `DecisionActionPanel`
- `GovernanceCopilotPanel`
- `GovernanceAuditTrail`

## Monitoring/Drift
- `MonitoringKPIRow`
- `DriftAlertList`
- `FeatureDriftHeatmap`
- `PerformanceTrendChart`
- `SegmentSliceTable`
- `RetrainDecisionCard`
- `MonitoringCopilotPanel`

Shared:
- `ArtifactDrawer`
- `GlobalContextStrip`
- `HealthStatusBadge`
- `EmptyState`
- `ErrorStateCard`

## 8) State Management Model

Server state:
- TanStack Query keyed by domain + params.
- Polling for batch jobs and selected monitoring tiles.

Client state:
- Local form state via React Hook Form.
- URL query params for filters/slices.
- Minimal global UI store (drawer open/close, theme, toasts).

Cache policy:
- Governance and monitoring: stale time 30-60s.
- Single valuation results: no stale reuse unless explicit compare mode.

## 9) Security and Compliance

- JWT + secure httpOnly cookie session in BFF.
- RBAC:
  - Analyst: valuation + batch read/write
  - Reviewer: governance read
  - Release owner: approve/reject
- Immutable audit events for governance actions.
- PII-safe logging: never log full address + all raw fields together.

## 10) Observability

Frontend:
- page load metrics, route transition timings, API latency by endpoint
- uncaught exceptions and rejected promises to error sink

BFF:
- structured logs with `request_id`, `user_id`, `route`, `status_code`
- traces for downstream backend calls

Operational dashboards:
- p95 latency by page/endpoint
- error rate by domain (valuation/governance/monitoring)

## 11) Implementation Phasing

## Phase 1: Platform scaffold
Acceptance:
1. App shell with auth and page routing exists.
2. Design tokens are live through CSS vars and DS primitives.
3. CI runs lint/typecheck/unit tests.

## Phase 2: Valuation domain
Acceptance:
1. Single valuation end-to-end with evidence panel.
2. Batch upload + job status + download flow works.
3. Contract tests validate `/api/v1/valuations/*` schemas.
4. Property-level SHAP drivers and confidence caveats render from API.

## Phase 3: Governance + monitoring
Acceptance:
1. Governance page renders proposal gates and decisions.
2. Monitoring page shows drift/performance slice diagnostics.
3. Artifact drawer deep-links to report paths.
4. Copilot endpoint integration provides grounded responses with citations.

## Phase 4: Hardening
Acceptance:
1. Playwright E2E covers critical flows.
2. SLO dashboards and alerting configured.
3. Accessibility audit meets WCAG AA for critical paths.

## 12) Open Questions

1. Should valuation API support synchronous and async single-value modes?
2. Is governance approval single-actor or dual-approval in v1?
3. Should artifact downloads be proxied by BFF or direct signed URLs?

## 13) Decision Checklist

Before implementation kickoff:
1. Approve this RFC.
2. Freeze v1 API DTO schemas.
3. Confirm auth provider and RBAC mapping.
4. Confirm design-system package ownership and release process.
