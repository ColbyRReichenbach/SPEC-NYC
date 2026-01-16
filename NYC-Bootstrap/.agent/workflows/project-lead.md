---
description: Master orchestrator - routes tasks to appropriate workflows
---

# Project Lead Workflow

**Role**: Orchestrate project execution, enforce phase gating, and route tasks to specialized workflows.

---

## Before Any Work

// turbo
1. Read the current project state:
   ```
   View file: .agent/workflows/context.md
   ```

2. Check the implementation plan status:
   ```
   View file: docs/NYC_IMPLEMENTATION_PLAN.md
   ```

3. Identify the current phase and next uncompleted task

---

## Task Routing Table

Use this table to direct work to the appropriate workflow:

| Request Contains | Route To | Workflow File |
|-----------------|----------|---------------|
| "setup", "docker", "infrastructure", "postgres" | Data Engineer | `/data-engineer` |
| "ETL", "data", "BBL", "cleaning", "ingest" | Data Engineer | `/data-engineer` |
| "schema", "database", "tables" | Data Engineer | `/data-engineer` |
| "model", "training", "XGBoost", "Optuna" | ML Engineer | `/ml-engineer` |
| "SHAP", "explain", "features" | ML Engineer | `/ml-engineer` |
| "quantile", "uncertainty", "confidence" | ML Engineer | `/ml-engineer` |
| "LLM", "prompt", "token", "OpenAI", "memo" | AI Security | `/ai-security` |
| "API", "FastAPI", "endpoint" | Full-Stack (V3.0+) | `/full-stack` |
| "frontend", "React", "UI" | Full-Stack (V3.0+) | `/full-stack` |
| "test", "validate", "verify" | Validation | `/validate` |

---

## Phase Gating Rules

### V1.0 Gate (Must complete before V2.0)
- [ ] Docker Compose runs successfully
- [ ] PostgreSQL contains ≥50,000 cleaned records
- [ ] XGBoost model achieves ≥70% PPE10
- [ ] SHAP explanations render correctly
- [ ] Git tagged as `v1.0`

### V2.0 Gate (Must complete before V3.0)
- [ ] Quantile regression implemented
- [ ] Confidence intervals displayed
- [ ] Backtest achieves ≥70% PPE10
- [ ] Git tagged as `v2.0`

### V3.0 Gate (Must complete before V4.0)
- [ ] FastAPI backend operational
- [ ] React frontend functional
- [ ] API latency <500ms
- [ ] Git tagged as `v3.0`

---

## Scope Enforcement

**ALWAYS DEFER these features** (from Implementation Plan):
- User authentication
- Multi-city support
- Real-time data feeds
- Mobile app
- USPAP compliance

If the user requests a deferred feature, respond:
> "This feature is marked as deferred in the Implementation Plan. I recommend completing V1.0 first to prove the core value. Should I add this to a future milestone instead?"

---

## After Task Completion

1. Update `context.md`:
   - Mark task as `complete`
   - Add entry to "Recent Changes"
   - Update "Next Action"

2. Update `NYC_IMPLEMENTATION_PLAN.md`:
   - Check off completed items with `[x]`

3. Commit changes:
   ```bash
   git add -A
   git commit -m "Complete [task name] - Phase [X.X]"
   ```

4. If phase complete, suggest next phase and appropriate workflow

---

## Emergency Protocols

### If Build Fails
Route to `/validate` workflow with error logs

### If Model Performance Below Target
1. Check data quality first (`/data-engineer` validation)
2. Review feature engineering
3. Consider hyperparameter re-tuning

### If Blocked by External Dependency
1. Add to "Blocking Issues" in `context.md`
2. Suggest workaround or mock data approach
3. Document in README

---

## Quick Status Check

// turbo
Run this to verify project health:
```bash
cd /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap && \
docker-compose ps 2>/dev/null || echo "Docker not running" && \
ls -la data/raw/ 2>/dev/null || echo "No raw data yet" && \
ls -la models/ 2>/dev/null || echo "No models yet"
```
