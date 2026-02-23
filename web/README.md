# Web App (Monorepo Frontend)

This directory contains the production AVM dashboard scaffold (Next.js App Router + BFF route handlers).

## What is included

- Dashboard page routes:
  - `/valuation/single`
  - `/valuation/batch`
  - `/governance`
  - `/monitoring`
  - `/copilot`
  - `/artifacts`
- API route scaffolds under `/api/v1/*`
- Typed contract schemas using `zod`
- Contract tests with `vitest`

## Prerequisites

- Node.js 20+
- npm 10+

## Install

Run from repository root:

```bash
npm install
```

## Development

From root:

```bash
npm run -w web dev
```

App runs on `http://localhost:3000` by default.

## Quality checks

From root:

```bash
npm run frontend:lint
npm run frontend:typecheck
npm run frontend:test:contracts
```

Or run all:

```bash
npm run frontend:ci
```

## API scaffolds

Current route handlers are typed scaffolds and return deterministic placeholder payloads for frontend integration.

Primary paths:

- `POST /api/v1/valuations/single`
- `POST /api/v1/valuations/batch`
- `GET /api/v1/valuations/batch/{jobId}`
- `GET /api/v1/governance/status`
- `POST /api/v1/governance/proposals/generate`
- `GET /api/v1/governance/proposals/latest`
- `POST /api/v1/governance/proposals/{proposalId}/approve`
- `POST /api/v1/governance/proposals/{proposalId}/reject`
- `GET /api/v1/monitoring/overview`
- `GET /api/v1/monitoring/drift`
- `GET /api/v1/monitoring/performance`
- `GET /api/v1/monitoring/retrain-decision/latest`
- `POST /api/v1/explanations/property`
- `POST /api/v1/copilot/ask`

## Next implementation steps

1. Replace placeholder route responses with real backend client calls.
2. Hook each dashboard page to its route handlers via typed data hooks.
3. Wire authentication and role-based access controls.
4. Add Playwright flows for single valuation, batch valuation, and governance decisioning.
