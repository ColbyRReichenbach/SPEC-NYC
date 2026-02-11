#!/usr/bin/env bash

set -euo pipefail

START_YEAR=2019
END_YEAR=2025
SKIP_DOWNLOAD=0
SKIP_DB_START=0
DRY_RUN=0
REPLACE_SALES=1
WRITE_REPORT=1
STOP_DB_AFTER=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_data.sh [options]

Bootstraps real NYC data for S.P.E.C. NYC:
1) downloads Annualized Sales raw data
2) starts Postgres (optional)
3) runs ETL (dry-run or load)

Options:
  --start-year YEAR     First year to request from connector (default: 2019)
  --end-year YEAR       Last year to request from connector (default: 2025)
  --skip-download       Do not run connector; use existing raw file
  --skip-db-start       Do not run docker compose up -d db
  --dry-run             Run ETL without loading to Postgres
  --no-replace          Append on load (default is replace/truncate sales table)
  --no-report           Skip ETL markdown/csv report output
  --stop-db-after       Stop db service at end of script
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-year)
      START_YEAR="${2:?missing value for --start-year}"
      shift 2
      ;;
    --end-year)
      END_YEAR="${2:?missing value for --end-year}"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-db-start)
      SKIP_DB_START=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-replace)
      REPLACE_SALES=0
      shift
      ;;
    --no-report)
      WRITE_REPORT=0
      shift
      ;;
    --stop-db-after)
      STOP_DB_AFTER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RAW_FILE="data/raw/annualized_sales_${START_YEAR}_${END_YEAR}.csv"

cleanup() {
  if [[ "$STOP_DB_AFTER" -eq 1 ]]; then
    echo "Stopping db service..."
    docker compose stop db >/dev/null
  fi
}
trap cleanup EXIT

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  echo "Downloading raw data (${START_YEAR}-${END_YEAR})..."
  python3 -m src.connectors --start-year "$START_YEAR" --end-year "$END_YEAR"
else
  echo "Skipping download (--skip-download)."
fi

if [[ ! -f "$RAW_FILE" ]]; then
  echo "Raw file not found: $RAW_FILE" >&2
  echo "Run without --skip-download or choose years that match an existing raw file." >&2
  exit 1
fi

if [[ "$SKIP_DB_START" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
  echo "Starting db service..."
  docker compose up -d db

  DB_CONTAINER_ID="$(docker compose ps -q db)"
  if [[ -z "$DB_CONTAINER_ID" ]]; then
    echo "Failed to resolve db container id from docker compose." >&2
    exit 1
  fi

  echo "Waiting for db health..."
  for _ in {1..60}; do
    HEALTH="$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}' "$DB_CONTAINER_ID" 2>/dev/null || true)"
    if [[ "$HEALTH" == "healthy" || "$HEALTH" == "running" ]]; then
      break
    fi
    sleep 2
  done

  FINAL_HEALTH="$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}' "$DB_CONTAINER_ID" 2>/dev/null || true)"
  if [[ "$FINAL_HEALTH" != "healthy" && "$FINAL_HEALTH" != "running" ]]; then
    echo "DB did not become healthy/running. Current status: ${FINAL_HEALTH:-unknown}" >&2
    exit 1
  fi
elif [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Skipping db start (--skip-db-start). Assuming db is already available."
fi

ETL_CMD=(python3 -m src.etl --input "$RAW_FILE")
if [[ "$DRY_RUN" -eq 1 ]]; then
  ETL_CMD+=(--dry-run)
fi
if [[ "$REPLACE_SALES" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  ETL_CMD+=(--replace-sales)
fi
if [[ "$WRITE_REPORT" -eq 1 ]]; then
  ETL_CMD+=(--write-report)
fi

echo "Running ETL..."
"${ETL_CMD[@]}"

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Verifying loaded row count..."
  python3 - <<'PY'
from sqlalchemy import create_engine, text
from config.settings import DATABASE_URL

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    total = conn.execute(text("select count(*) from sales")).scalar()
    min_date = conn.execute(text("select min(sale_date) from sales")).scalar()
    max_date = conn.execute(text("select max(sale_date) from sales")).scalar()
print(f"sales rows={total}, date_range={min_date}..{max_date}")
PY
fi

echo "Bootstrap complete."
