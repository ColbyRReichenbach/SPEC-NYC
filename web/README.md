# Web App (Azuli-Branded, Canonical-Contract Frontend)

This directory contains the production AVM dashboard built on Next.js App Router + BFF.

## Architecture rule

The frontend and BFF must never depend on raw client database schemas.

- UI consumes only canonical `/api/v1` contracts.
- Unknown datasource mapping stays in backend ingestion/canonicalization.
- Source metadata is passed via `source_context` on API responses.

## Implemented v1 scope

- `Single Valuation` (guided + expert + map-based property explorer)
- `Governance` (read-only in no-auth demo mode)
- `Monitoring`
- Contextual copilot side panel with grounded citations
- SHAP visuals (local waterfall + global summary)

## Routes

Dashboard pages:

- `/valuation/single`
- `/governance`
- `/monitoring`
- `/artifacts`

API routes:

- `POST /api/v1/valuations/single`
- `POST /api/v1/explanations/property`
- `GET /api/v1/explanations/shap/global`
- `GET /api/v1/properties/search`
- `GET /api/v1/properties/nearby`
- `GET /api/v1/properties/{propertyId}`
- `GET /api/v1/governance/status`
- `GET /api/v1/governance/proposals/latest`
- `POST /api/v1/governance/proposals/{proposalId}/approve` (disabled/no-auth)
- `POST /api/v1/governance/proposals/{proposalId}/reject` (disabled/no-auth)
- `GET /api/v1/monitoring/overview`
- `GET /api/v1/monitoring/drift`
- `GET /api/v1/monitoring/performance`
- `GET /api/v1/monitoring/retrain-decision/latest`
- `POST /api/v1/copilot/ask`

## Contracts

All canonical API responses include:

- `request_id`
- `contract_version` (`v1`)
- `generated_at`
- `source_context`

## Run locally

From repository root:

```bash
npm install
npm run -w web dev
```

## Run with Docker Compose

From repository root:

```bash
docker compose up web
```

Frontend is served at `http://localhost:3000`.

## Checks

```bash
npm run frontend:lint
npm run frontend:typecheck
npm run frontend:test:contracts
# or
npm run frontend:ci
```
