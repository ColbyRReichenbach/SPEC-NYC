# Codex Control Plane (SPEC-NYC)

This directory turns SPEC-NYC into a Codex-first autonomous delivery system.

## Core invariant
Codex can iterate freely, but every loop must end with:

`python3 -m src.validate_release --mode smoke --contract-profile canonical`

No PR should be opened unless Gate E is green.

## State machine
- `DATA_OK`: Gate A passed (including canonicalization smoke)
- `MODEL_OK`: Gate B passed
- `PRODUCT_OK`: Gate C passed
- `OPS_OK`: Gate D passed
- `RELEASABLE`: Gate E passed

Use role cards in `agents/` and sequence in `workflows/`.


## Autopilot helper
Use `./scripts/autonomy_loop.sh` for iterative validate->repair->revalidate loops with a repair brief artifact at `.codex/runtime/repair_brief.json`.
