#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

allowed_paths=(
  "web/src/lib/brand.ts"
  "web/middleware.ts"
  "web/app/globals.css"
)

matches="$(rg --no-heading --line-number --color never -S '(Azuli|azuli)' web || true)"

if [[ -z "${matches}" ]]; then
  echo "Brand literal guard passed: no Azuli literals found."
  exit 0
fi

violations=()
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  path="${line%%:*}"
  is_allowed=false
  for allowed in "${allowed_paths[@]}"; do
    if [[ "${path}" == "${allowed}" ]]; then
      is_allowed=true
      break
    fi
  done
  if [[ "${is_allowed}" == false ]]; then
    violations+=("${line}")
  fi
done <<< "${matches}"

if (( ${#violations[@]} > 0 )); then
  echo "Brand literal guard failed. Found Azuli literal(s) outside allowed branding files:"
  for violation in "${violations[@]}"; do
    echo "  ${violation}"
  done
  echo
  echo "Allowed files:"
  for allowed in "${allowed_paths[@]}"; do
    echo "  ${allowed}"
  done
  exit 1
fi

echo "Brand literal guard passed: Azuli literals only exist in allowed branding files."
