# Workflow: Autopilot Loop (Low-HITL)

Use this when you want Codex to continuously validate and self-correct until gates are green.

## Command
`./scripts/autonomy_loop.sh`

## What it does
1. Runs `src.validate_release` in smoke mode with canonical contracts.
2. If blocked, writes a machine-readable repair brief to `.codex/runtime/repair_brief.json`.
3. Optionally runs a repair command and retries (up to `--max-iterations`).

## Example with a repair hook
`./scripts/autonomy_loop.sh --repair-command "PYTHONPATH=. pytest -q tests/test_etl.py tests/test_validate_release.py"`

## HITL boundary recommendation
- Keep human approval only at final PR/release.
- Let loops run autonomously for code/test/repair cycles.
