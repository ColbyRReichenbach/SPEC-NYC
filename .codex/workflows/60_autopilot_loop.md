# Workflow: Autopilot Loop (Low-HITL)

Use this when you want Codex to continuously validate and self-correct until gates are green.

## Command
`./scripts/autonomy_loop.sh`

## What it does
1. Runs `src.validate_release` in smoke mode with canonical contracts.
2. If blocked, writes a machine-readable repair brief to `.codex/runtime/repair_brief.json`, including gate-category diagnosis (`Data/Model/Product/Ops`) and check-level recommended actions.
3. Optionally runs a repair command and retries (up to `--max-iterations`).
4. Optionally runs a gate-aware repair command pack (`--repair-command-pack <json>`), keyed by gate name (for example `Gate A (Data)`) or category (for example `Data`).

## Example with a repair hook
`./scripts/autonomy_loop.sh --repair-command "PYTHONPATH=. pytest -q tests/test_etl.py tests/test_validate_release.py"`

## Example with gate-aware repair pack
`./scripts/autonomy_loop.sh --repair-command-pack config/autonomy/repair_pack.example.json`

## HITL boundary recommendation
- Keep human approval only at final PR/release.
- Let loops run autonomously for code/test/repair cycles.
