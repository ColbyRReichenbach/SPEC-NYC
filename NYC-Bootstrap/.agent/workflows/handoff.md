---
description: Protocol for transitioning between workflow phases
---

# Handoff Protocol

**Purpose**: Minimal overhead transitions between phases.

---

## The Simple Rule

> **The implementation plan is the tracker. Check off items. That's it.**

---

## When You Complete Work

### 1. Edit the Implementation Plan
Open `docs/NYC_IMPLEMENTATION_PLAN.md` and change:
```markdown
- [ ] Task you just did
```
to:
```markdown
- [x] Task you just did
```

### 2. Update State (Only If Changing Phase)
If moving to a new phase, edit `.agent/workflows/state.yaml`:
```yaml
current_phase: "1.4"
phase_name: "Data Ingestion"
status: "in_progress"
last_updated: "2026-01-16"
last_milestone: "Database schema created"
```

### 3. Commit
```bash
git add -A
git commit -m "[Phase X.X] Complete: brief description"
```

---

## Handoff Message Template

When transitioning to another workflow, say:

> **Handoff to `/[workflow]`**
> - Completed: [what you did]
> - Next step: [first unchecked item in plan]
> - Blockers: [none / issue]

That's the entire protocol.

---

## Quick Handoff Commands

// turbo
### See what's next
```bash
grep -n "^\s*- \[ \]" /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/docs/NYC_IMPLEMENTATION_PLAN.md | head -3
```

// turbo
### See recent completions
```bash
git log --oneline -5
```

---

## Recovery (If State Is Unclear)

If you're unsure where things stand:

1. Check git history:
   ```bash
   git log --oneline -10
   ```

2. Find first unchecked task:
   ```bash
   grep -n "^\s*- \[ \]" docs/NYC_IMPLEMENTATION_PLAN.md | head -1
   ```

3. Read state file:
   ```bash
   cat .agent/workflows/state.yaml
   ```

4. If state.yaml is wrong, update it based on the plan checkboxes.

---

## Why This Works

- **One source of truth**: The plan's checkboxes
- **Minimal state file**: Just a pointer for quick reference
- **Git is the changelog**: Commit messages document progress
- **No tables to maintain**: Checkboxes are the tables
