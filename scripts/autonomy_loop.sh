#!/usr/bin/env bash
set -euo pipefail

python3 -m src.autonomy.loop_runner \
  --mode smoke \
  --contract-profile canonical \
  --data-source csv \
  --mapping-yaml src/datasources/mappings/spec_nyc_v1.yaml \
  --max-iterations 3 \
  "$@"
