# Hypothesis Backlog and Branching Plan

Purpose:
- Keep a single list of active model ideas.
- Track what we test next and why.
- Prevent confusion when multiple challengers branch from different baselines.

---

## 1) How to Think About This

Plain version:
1. The `champion` is production truth.
2. Every new idea is a `candidate`.
3. Some candidates fail promotion but still teach us something useful.
4. We are allowed to branch a new idea from a non-champion candidate for research.
5. Promotion to production always requires beating the current champion in arena gates.

This is normal in real production teams. It is not a mistake.

---

## 2) Current Planned Ideas

| Hypothesis ID | Parent Hypothesis | Main Change | Why Test It | Expected Gain | Status |
|---|---|---|---|---|---|
| `H-SEG-001` | `champion` | Segment-only router (`property_segment`) | Global model appears uneven by property type | Better segment stability + lower weighted MdAPE | ready |
| `H-SEG-TIER-001` | `H-SEG-001` | Segment + tier router (`property_segment + price_tier_proxy`) | High/low value dynamics may differ within segment | Additional slice accuracy uplift | planned |
| `H-SEASON-001` | `H-SEG-001` | Add seasonality features and rolling-time validation | Real estate is regime/season sensitive | Better temporal stability and fewer drift failures | planned |
| `H-SEASON-TIER-001` | `H-SEG-TIER-001` | Segment+tier + seasonal features | Combined architecture if both ideas work | Best overall/slice performance if data volume supports | planned |

Note:
- Parent hypothesis is an experiment lineage pointer, not a production promotion rule.
- Tier routing must use non-leaky proxy tier (`price_tier_proxy`), not target-derived `price_tier`.

---

## 3) Branching Rule (Important)

If `H-SEG-TIER-001` does not become champion, can we still test `H-SEASON-TIER-001`?
- Yes, for research and learning.
- But final promotion must compare candidate against current champion.

So we keep two judgments:
1. Local comparison: candidate vs its parent (did this new idea help that branch?).
2. Production comparison: candidate vs champion (is it safe and better for production?).

Only #2 can change champion alias.

---

## 4) Decision Policy

1. Default path:
- Start new hypotheses from current champion behavior.

2. Allowed exception:
- Branch from a non-champion candidate when that candidate introduced a strong mechanism worth extending.

3. Promotion gate:
- Arena proposal must pass policy thresholds in `config/arena_policy.yaml`.

4. Tie-break:
- Keep the path that is simpler and more stable if gains are similar.

---

## 5) Execution Sequence for Current Backlog

1. Run `H-SEG-001` (segment-only router).
2. Decision point:
- If strong uplift and clean gates: branch `H-SEG-TIER-001` and `H-SEASON-001`.
- If weak uplift: prioritize `H-SEASON-001` off champion/global behavior.
3. Run `H-SEG-TIER-001`.
4. Run `H-SEASON-001`.
5. Optional combined run `H-SEASON-TIER-001` only if both individual effects are promising.
6. Compare finalists against champion in arena and decide.

---

## 6) Required Logging for Each Hypothesis

For every hypothesis run, record:
1. `hypothesis_id`
2. `change_type`
3. `change_summary`
4. `feature_set_version`
5. `dataset_version`
6. `owner`
7. `parent_hypothesis_id` (manual note in hypothesis file if not in CLI tags yet)
8. `base_alias_snapshot` (which champion version existed when run started)

This keeps experiment lineage auditable.

---

## 7) Quick Summary

If you remember one thing:
- "Many ideas can be explored, but only ideas that beat champion with guardrails can ship."

That is exactly how real DS teams avoid both stagnation and chaos.
