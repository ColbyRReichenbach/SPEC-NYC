#!/usr/bin/env bash

set -euo pipefail

TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
EXPERIMENT_NAME="spec-nyc-avm"
REGISTERED_MODEL_NAME="spec-nyc-avm"
ARENA_POLICY_PATH="config/arena_policy.yaml"
ARENA_DIR="reports/arena"

usage() {
  cat <<'EOF'
Usage: scripts/ds_workflow.sh <command> [options]

Commands:
  daily
      Run a realistic daily DS check loop:
      - core test suite
      - arena lifecycle status
      - monitoring checks (when inputs exist)

  train-candidate [options]
      Train a challenger candidate and register it to MLflow with alias=candidate.
      Required options:
        --hypothesis-id <id>
        --change-type <feature|objective|data|architecture|tuning>
        --change-summary <text>
        --owner <name>
        --feature-set-version <version>
        --dataset-version <version>
      Optional options:
        --input-csv <path>
        --limit <int>
        --test-size <float>        (default: 0.2)
        --optuna-trials <int>      (default: 30)
        --strategy <name>          (default: global; options: global, segmented_router)
        --min-segment-rows <int>   (default: 2000; segmented_router only)
        --router-mode <name>       (default: segment_only; options: segment_only, segment_plus_tier)
        --model-version <name>     (default: v1)
        --artifact-tag <tag>       (default: auto candidate_<timestamp>)
        --run-name <name>          (default: auto)
        --shap-sample-size <int>   (default: 500)

  arena-propose
      Generate a champion/challenger proposal from recent eligible candidates.

  bootstrap-champion [options]
      One-time command to seed the first champion alias from existing artifacts.
      Optional options:
        --model-version <name>     (default: v1)
        --artifact-tag <tag>       (default: prod)
        --dataset-version <value>  (default: bootstrap_<timestamp>)
        --owner <name>             (default: $USER)
        --run-name <name>          (default: bootstrap-<model>-<tag>)

  arena-status
      Show registry aliases and latest proposal state.

  arena-approve --proposal-id <id> --approved-by <name>
      Approve proposal and atomically promote champion alias.

  arena-reject --proposal-id <id> --reason <text> --rejected-by <name>
      Reject proposal; champion remains unchanged.

  release-check
      Run production release validation (includes arena governance checks).

  mlflow-ui [--port <port>]
      Launch MLflow UI for experiment comparison.
EOF
}

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "Missing required option: ${name}" >&2
    exit 2
  fi
}

run_daily() {
  echo "== Daily DS checks =="
  python3 -m unittest tests.test_etl tests.test_evaluate tests.test_monitoring tests.test_arena -q

  echo
  echo "== Arena status =="
  python3 -m src.mlops.arena status \
    --tracking-uri "${TRACKING_URI}" \
    --policy-path "${ARENA_POLICY_PATH}" || true

  echo
  echo "== Monitoring =="
  if [[ -f "reports/monitoring/reference_slice_v1.csv" && -f "reports/monitoring/current_slice_v1.csv" ]]; then
    python3 -m src.monitoring.drift \
      --reference-csv reports/monitoring/reference_slice_v1.csv \
      --current-csv reports/monitoring/current_slice_v1.csv
  else
    echo "Skipping drift monitor: reference/current slices not found."
  fi

  if [[ -f "reports/model/evaluation_predictions_v1.csv" ]]; then
    python3 -m src.monitoring.performance \
      --predictions-csv reports/model/evaluation_predictions_v1.csv
  else
    echo "Skipping performance monitor: reports/model/evaluation_predictions_v1.csv not found."
  fi

  python3 -m src.retrain_policy
}

run_train_candidate() {
  local hypothesis_id=""
  local change_type=""
  local change_summary=""
  local owner=""
  local feature_set_version=""
  local dataset_version=""
  local input_csv=""
  local limit=""
  local test_size="0.2"
  local optuna_trials="30"
  local strategy="global"
  local min_segment_rows="2000"
  local router_mode="segment_only"
  local model_version="v1"
  local artifact_tag=""
  local run_name=""
  local shap_sample_size="500"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --hypothesis-id) hypothesis_id="${2:?missing value for --hypothesis-id}"; shift 2 ;;
      --change-type) change_type="${2:?missing value for --change-type}"; shift 2 ;;
      --change-summary) change_summary="${2:?missing value for --change-summary}"; shift 2 ;;
      --owner) owner="${2:?missing value for --owner}"; shift 2 ;;
      --feature-set-version) feature_set_version="${2:?missing value for --feature-set-version}"; shift 2 ;;
      --dataset-version) dataset_version="${2:?missing value for --dataset-version}"; shift 2 ;;
      --input-csv) input_csv="${2:?missing value for --input-csv}"; shift 2 ;;
      --limit) limit="${2:?missing value for --limit}"; shift 2 ;;
      --test-size) test_size="${2:?missing value for --test-size}"; shift 2 ;;
      --optuna-trials) optuna_trials="${2:?missing value for --optuna-trials}"; shift 2 ;;
      --strategy) strategy="${2:?missing value for --strategy}"; shift 2 ;;
      --min-segment-rows) min_segment_rows="${2:?missing value for --min-segment-rows}"; shift 2 ;;
      --router-mode) router_mode="${2:?missing value for --router-mode}"; shift 2 ;;
      --model-version) model_version="${2:?missing value for --model-version}"; shift 2 ;;
      --artifact-tag) artifact_tag="${2:?missing value for --artifact-tag}"; shift 2 ;;
      --run-name) run_name="${2:?missing value for --run-name}"; shift 2 ;;
      --shap-sample-size) shap_sample_size="${2:?missing value for --shap-sample-size}"; shift 2 ;;
      *)
        echo "Unknown option for train-candidate: $1" >&2
        exit 2
        ;;
    esac
  done

  require_value "--hypothesis-id" "${hypothesis_id}"
  require_value "--change-type" "${change_type}"
  require_value "--change-summary" "${change_summary}"
  require_value "--owner" "${owner}"
  require_value "--feature-set-version" "${feature_set_version}"
  require_value "--dataset-version" "${dataset_version}"

  if [[ -z "${artifact_tag}" ]]; then
    artifact_tag="candidate_$(date -u +%Y%m%d_%H%M%S)"
  fi
  if [[ -z "${run_name}" ]]; then
    run_name="train-${model_version}-${artifact_tag}"
  fi

  local stem="${model_version}"
  if [[ "${artifact_tag}" != "prod" ]]; then
    stem="${model_version}_${artifact_tag}"
  fi

  local model_path="models/model_${stem}.joblib"
  local metrics_path="models/metrics_${stem}.json"
  local scorecard_path="reports/model/segment_scorecard_${stem}.csv"
  local predictions_path="reports/model/evaluation_predictions_${stem}.csv"

  echo "== Training candidate model =="
  local model_cmd=(
    python3 -m src.model
    --model-version "${model_version}"
    --artifact-tag "${artifact_tag}"
    --strategy "${strategy}"
    --min-segment-rows "${min_segment_rows}"
    --router-mode "${router_mode}"
    --test-size "${test_size}"
    --optuna-trials "${optuna_trials}"
    --shap-sample-size "${shap_sample_size}"
    --dataset-version "${dataset_version}"
    --tracking-uri "${TRACKING_URI}"
    --no-mlflow
  )
  if [[ -n "${input_csv}" ]]; then
    model_cmd+=(--input-csv "${input_csv}")
  fi
  if [[ -n "${limit}" ]]; then
    model_cmd+=(--limit "${limit}")
  fi
  "${model_cmd[@]}"

  if [[ ! -f "${model_path}" || ! -f "${metrics_path}" || ! -f "${scorecard_path}" || ! -f "${predictions_path}" ]]; then
    echo "Expected artifacts missing after training candidate." >&2
    echo "model=${model_path}" >&2
    echo "metrics=${metrics_path}" >&2
    echo "scorecard=${scorecard_path}" >&2
    echo "predictions=${predictions_path}" >&2
    exit 1
  fi

  echo
  echo "== Registering candidate to MLflow model registry =="
  python3 -m src.mlops.track_run \
    --metrics-json "${metrics_path}" \
    --model-artifact "${model_path}" \
    --scorecard-csv "${scorecard_path}" \
    --predictions-csv "${predictions_path}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --run-name "${run_name}" \
    --dataset-version "${dataset_version}" \
    --tracking-uri "${TRACKING_URI}" \
    --hypothesis-id "${hypothesis_id}" \
    --change-type "${change_type}" \
    --change-summary "${change_summary}" \
    --owner "${owner}" \
    --feature-set-version "${feature_set_version}" \
    --register-model \
    --registered-model-name "${REGISTERED_MODEL_NAME}" \
    --alias candidate \
    --run-kind train \
    --arena-dir "${ARENA_DIR}"
}

run_arena_propose() {
  python3 -m src.mlops.arena propose \
    --tracking-uri "${TRACKING_URI}" \
    --policy-path "${ARENA_POLICY_PATH}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --arena-dir "${ARENA_DIR}"
}

run_bootstrap_champion() {
  local model_version="v1"
  local artifact_tag="prod"
  local dataset_version="bootstrap_$(date -u +%Y%m%d_%H%M%S)"
  local owner="${USER:-unknown}"
  local run_name=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-version) model_version="${2:?missing value for --model-version}"; shift 2 ;;
      --artifact-tag) artifact_tag="${2:?missing value for --artifact-tag}"; shift 2 ;;
      --dataset-version) dataset_version="${2:?missing value for --dataset-version}"; shift 2 ;;
      --owner) owner="${2:?missing value for --owner}"; shift 2 ;;
      --run-name) run_name="${2:?missing value for --run-name}"; shift 2 ;;
      *)
        echo "Unknown option for bootstrap-champion: $1" >&2
        exit 2
        ;;
    esac
  done

  local stem="${model_version}"
  if [[ "${artifact_tag}" != "prod" ]]; then
    stem="${model_version}_${artifact_tag}"
  fi
  local model_path="models/model_${stem}.joblib"
  local metrics_path="models/metrics_${stem}.json"
  local scorecard_path="reports/model/segment_scorecard_${stem}.csv"
  local predictions_path="reports/model/evaluation_predictions_${stem}.csv"

  if [[ -z "${run_name}" ]]; then
    run_name="bootstrap-${model_version}-${artifact_tag}"
  fi

  if [[ ! -f "${model_path}" || ! -f "${metrics_path}" ]]; then
    echo "Bootstrap artifacts missing. Expected at least:" >&2
    echo "  ${model_path}" >&2
    echo "  ${metrics_path}" >&2
    exit 1
  fi

  python3 -m src.mlops.track_run \
    --metrics-json "${metrics_path}" \
    --model-artifact "${model_path}" \
    --scorecard-csv "${scorecard_path}" \
    --predictions-csv "${predictions_path}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --run-name "${run_name}" \
    --dataset-version "${dataset_version}" \
    --tracking-uri "${TRACKING_URI}" \
    --hypothesis-id "H-000-BOOTSTRAP" \
    --change-type data \
    --change-summary "Initial champion baseline bootstrap" \
    --owner "${owner}" \
    --feature-set-version "${model_version}" \
    --register-model \
    --registered-model-name "${REGISTERED_MODEL_NAME}" \
    --alias champion \
    --run-kind train \
    --arena-dir "${ARENA_DIR}"
}

run_arena_status() {
  python3 -m src.mlops.arena status \
    --tracking-uri "${TRACKING_URI}" \
    --policy-path "${ARENA_POLICY_PATH}" \
    --arena-dir "${ARENA_DIR}"
}

run_arena_approve() {
  local proposal_id=""
  local approved_by=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --proposal-id) proposal_id="${2:?missing value for --proposal-id}"; shift 2 ;;
      --approved-by) approved_by="${2:?missing value for --approved-by}"; shift 2 ;;
      *)
        echo "Unknown option for arena-approve: $1" >&2
        exit 2
        ;;
    esac
  done
  require_value "--proposal-id" "${proposal_id}"
  require_value "--approved-by" "${approved_by}"
  python3 -m src.mlops.arena approve \
    --proposal-id "${proposal_id}" \
    --tracking-uri "${TRACKING_URI}" \
    --policy-path "${ARENA_POLICY_PATH}" \
    --arena-dir "${ARENA_DIR}" \
    --approved-by "${approved_by}"
}

run_arena_reject() {
  local proposal_id=""
  local reason=""
  local rejected_by=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --proposal-id) proposal_id="${2:?missing value for --proposal-id}"; shift 2 ;;
      --reason) reason="${2:?missing value for --reason}"; shift 2 ;;
      --rejected-by) rejected_by="${2:?missing value for --rejected-by}"; shift 2 ;;
      *)
        echo "Unknown option for arena-reject: $1" >&2
        exit 2
        ;;
    esac
  done
  require_value "--proposal-id" "${proposal_id}"
  require_value "--reason" "${reason}"
  require_value "--rejected-by" "${rejected_by}"
  python3 -m src.mlops.arena reject \
    --proposal-id "${proposal_id}" \
    --tracking-uri "${TRACKING_URI}" \
    --policy-path "${ARENA_POLICY_PATH}" \
    --arena-dir "${ARENA_DIR}" \
    --reason "${reason}" \
    --rejected-by "${rejected_by}"
}

run_release_check() {
  python3 -m src.validate_release \
    --mode production \
    --mlflow-tracking-uri "${TRACKING_URI}" \
    --arena-policy-path "${ARENA_POLICY_PATH}" \
    --arena-dir "${ARENA_DIR}"
}

run_mlflow_ui() {
  local port="5001"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port) port="${2:?missing value for --port}"; shift 2 ;;
      *)
        echo "Unknown option for mlflow-ui: $1" >&2
        exit 2
        ;;
    esac
  done

  python3 -m mlflow ui \
    --backend-store-uri "${TRACKING_URI}" \
    --host 127.0.0.1 \
    --port "${port}"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local command="$1"
  shift
  case "${command}" in
    daily) run_daily "$@" ;;
    train-candidate) run_train_candidate "$@" ;;
    arena-propose) run_arena_propose "$@" ;;
    bootstrap-champion) run_bootstrap_champion "$@" ;;
    arena-status) run_arena_status "$@" ;;
    arena-approve) run_arena_approve "$@" ;;
    arena-reject) run_arena_reject "$@" ;;
    release-check) run_release_check "$@" ;;
    mlflow-ui) run_mlflow_ui "$@" ;;
    -h|--help|help) usage ;;
    *)
      echo "Unknown command: ${command}" >&2
      usage
      exit 2
      ;;
  esac
}

main "$@"
