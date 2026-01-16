---
description: Protocol for transitioning between workflow phases
---

# Handoff Protocol

**Purpose**: Ensure clean transitions between workflow phases with proper documentation and state updates.

---

## When to Use This Workflow

Use this protocol when:
- Completing a major phase (e.g., Data Engineering → Model Training)
- Switching between personas/workflows
- Handing off to a different agent session
- Resuming work after a break

---

## Handoff Checklist

### 1. Document Current State

Update `.agent/workflows/context.md`:

```markdown
## Recent Changes
| Date | Change | Persona |
|------|--------|---------|
| [TODAY] | [What was completed] | [Which workflow] |
```

### 2. Update Implementation Plan

In `docs/NYC_IMPLEMENTATION_PLAN.md`, mark completed items:

```markdown
- [x] Task that was completed
- [ ] Next task to do
```

### 3. Commit Changes

```bash
git add -A
git status  # Review what's being committed
git commit -m "[Phase X.X] Complete [task description]

- Completed: [bullet list of done items]
- Metrics: [if applicable]
- Next: [what comes next]"
```

### 4. Identify Next Workflow

Check the routing table in `/project-lead`:

| Current Phase Complete | Next Workflow |
|-----------------------|---------------|
| Infrastructure Setup | `/data-engineer` (Data Ingestion) |
| Data Cleaning | `/data-engineer` (Feature Engineering) |
| Feature Engineering | `/ml-engineer` (Model Training) |
| Model Training | `/validate` (Verify metrics) |
| Validation Pass | `/project-lead` (Next phase) |

### 5. Write Handoff Summary

Create a clear summary for the next workflow:

```markdown
## Handoff Summary

**From**: [Current Workflow]
**To**: [Next Workflow]
**Date**: [Current Date]

### Completed
- [List of completed items]

### Artifacts Created
- [Files created or modified]

### Current Metrics
- PPE10: [value or "pending"]
- MdAPE: [value or "pending"]

### Next Steps
1. [First action for next workflow]
2. [Second action]

### Blocking Issues
- [Any issues or "None"]

### Notes for Next Session
- [Any context that would be helpful]
```

---

## Quick Handoff Commands

### Check What Changed
```bash
git diff --stat HEAD~1
```

### View Recent Commits
```bash
git log --oneline -5
```

### Verify No Uncommitted Work
```bash
git status
```

### Tag a Version
```bash
git tag -a v1.0 -m "V1.0: Core AVM with XGBoost baseline"
git push origin v1.0
```

---

## Handoff Templates

### Data Engineering → ML Engineering

```markdown
## Handoff: Data → ML

**Status**: Data pipeline complete ✓

### Database State
- Records in `sales`: [count]
- Records in `properties`: [count]
- Records in `features`: [count]

### Data Quality
- Zero sales removed: ✓
- Non-market filtered: ✓ (<$10k)
- Residential only: ✓ (A,B,C,D,R)
- BBL validated: ✓ (10-char)
- Sqft imputed: ✓

### Features Available
- sqft, year_built, units_total
- building_class, borough
- distance_to_center_km
- h3_index, h3_price_lag

### Ready For
- XGBoost training with Optuna
- Target: PPE10 ≥70%
```

### ML Engineering → Validation

```markdown
## Handoff: ML → Validation

**Status**: Model training complete ✓

### Model Artifacts
- `models/xgb_v1.joblib` - Trained model
- `models/metrics_v1.json` - Performance metrics
- `models/shap_waterfall_sample.png` - Explanation

### Metrics Achieved
- PPE10: [X]% (target: ≥70%)
- MdAPE: [X]% (target: ≤8%)
- R²: [X] (target: ≥0.75)

### Hyperparameters
- max_depth: [value]
- learning_rate: [value]
- n_estimators: [value]

### Ready For
- Full validation suite
- Dashboard integration
- V1.0 tag if passing
```

### Validation → Project Lead

```markdown
## Handoff: Validation → Project Lead

**Status**: Phase [X] validated ✓

### Checks Passed
- [ ] Docker services running
- [ ] Database populated
- [ ] Model metrics meet targets
- [ ] SHAP explanations work
- [ ] No security issues

### Checks Failed
- [List any failures, or "None"]

### Recommendation
- [Proceed to Phase X+1 / Fix issues / Tag release]
```

---

## Recovery Protocol

If handoff state is unclear:

1. **Read Context First**
   ```bash
   cat .agent/workflows/context.md
   ```

2. **Check Implementation Plan**
   ```bash
   grep -E "^\s*- \[[ x]\]" docs/NYC_IMPLEMENTATION_PLAN.md | head -20
   ```

3. **Review Recent Git History**
   ```bash
   git log --oneline -10
   ```

4. **Run Validation**
   Follow `/validate` workflow to assess current state

5. **Update Context**
   If context is stale, update it based on findings

---

## Best Practices

1. **Always commit before handoff** - Never leave work uncommitted
2. **Update context.md immediately** - Don't defer documentation
3. **Be specific about next steps** - Vague handoffs cause confusion
4. **Include metrics** - Numbers help track progress
5. **Note blockers** - Flag issues before they become surprises
