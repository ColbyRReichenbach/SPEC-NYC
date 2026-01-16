---
description: Master orchestrator - routes tasks based on implementation plan progress
---

# Project Lead Workflow

**Role**: Orchestrate project execution by reading implementation plan checkboxes and routing to the appropriate workflow.

---

## How This Works

1. **Single Source of Truth**: `docs/NYC_IMPLEMENTATION_PLAN.md` checkboxes
2. **Minimal State**: `.agent/workflows/state.yaml` (just a pointer)
3. **No Manual Tracking**: The plan IS the tracker

---

## Before Any Work

// turbo
1. Read current state:
   ```bash
   cat /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/.agent/workflows/state.yaml
   ```

// turbo
2. Find next incomplete task in the implementation plan:
   ```bash
   grep -n "^\s*- \[ \]" /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/docs/NYC_IMPLEMENTATION_PLAN.md | head -5
   ```

3. Identify which phase that task belongs to and route accordingly.

---

## Task Routing Table

| Task Keywords | Phase | Route To |
|--------------|-------|----------|
| "docker", "compose", "postgres", "infrastructure" | 1.2 | `/data-engineer` |
| "database", "schema", "SQLAlchemy", "tables" | 1.3 | `/data-engineer` |
| "download", "ingest", "BBL", "connectors" | 1.4 | `/data-engineer` |
| "clean", "filter", "ETL", "impute" | 1.5 | `/data-engineer` |
| "spatial", "h3", "distance", "feature" | 1.6 | `/data-engineer` |
| "train", "XGBoost", "Optuna", "model" | 1.7 | `/ml-engineer` |
| "dashboard", "streamlit", "app" | 1.8 | `/data-engineer` |
| "quantile", "confidence", "interval" | 2.1 | `/ml-engineer` |
| "subway", "MTA", "flood" | 2.2 | `/data-engineer` |
| "backtest", "holdout" | 2.3 | `/ml-engineer` |
| "comps", "comparable", "similar" | 2.4 | `/ml-engineer` |
| "FastAPI", "endpoint", "API" | 3.x | `/full-stack` |
| "React", "frontend", "Next.js" | 3.x | `/full-stack` |
| "LLM", "memo", "oracle", "OpenAI" | 4.x | `/ai-security` |
| "RAG", "ChromaDB", "vector" | 4.x | `/ai-security` |

---

## Phase Gating

### Before V2.0 (Check ALL)
```bash
# Run this to check V1.0 readiness
grep -E "^\s*- \[x\]" /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/docs/NYC_IMPLEMENTATION_PLAN.md | grep -E "(Docker Compose|PostgreSQL|Model achieves|SHAP|Map shows|README|Git tag.*v1)" | wc -l
```
Must return 7 (all V1.0 deliverables checked).

### Before V3.0 (Check ALL)
V2.0 deliverables section must be complete.

---

## Completing a Task

When a task is done:

1. **Check it off in the plan** (edit NYC_IMPLEMENTATION_PLAN.md):
   ```markdown
   - [x] Task that was completed  ← Change [ ] to [x]
   ```

2. **Update state.yaml** (only if phase changes):
   ```yaml
   current_phase: "1.3"  ← Increment if moving to new phase
   status: "in_progress"
   last_updated: "2026-01-16"
   last_milestone: "Infrastructure setup complete"
   ```

3. **Commit**:
   ```bash
   git add docs/NYC_IMPLEMENTATION_PLAN.md .agent/workflows/state.yaml
   git commit -m "Complete [task] - Phase [X.X]"
   ```

That's it. No tables to maintain.

---

## Scope Enforcement

**ALWAYS DEFER these features** (from Implementation Plan):
- User authentication
- Multi-city support  
- Real-time data feeds
- Mobile app
- USPAP compliance

If requested, respond:
> "This is marked as deferred. Let's complete V1.0 first—should I add this to a future milestone?"

---

## Quick Commands

### See all completed tasks
```bash
grep -c "^\s*- \[x\]" docs/NYC_IMPLEMENTATION_PLAN.md
```

### See all remaining tasks
```bash
grep -c "^\s*- \[ \]" docs/NYC_IMPLEMENTATION_PLAN.md
```

### Check current phase completion
```bash
# Count checked items in Phase 1
sed -n '/## Phase 1/,/## Phase 2/p' docs/NYC_IMPLEMENTATION_PLAN.md | grep -c "^\s*- \[x\]"
```

---

## When Stuck

1. Run `/validate` workflow to check project health
2. Check for blocking issues in `state.yaml`
3. Review the implementation plan for dependencies
4. If blocked by external factor, add to state.yaml:
   ```yaml
   blocking: "Waiting for NYC data portal access"
   ```
