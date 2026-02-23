from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


GATE_CATEGORY_MAP = {
    "Gate A (Data)": "Data",
    "Gate B (Model)": "Model",
    "Gate C (Product)": "Product",
    "Gate D (Ops)": "Ops",
    "Gate E (Release)": "Release",
}
GATE_PRIORITY = {name: idx for idx, name in enumerate(GATE_CATEGORY_MAP)}

CHECK_REPAIR_ACTIONS: dict[str, list[str]] = {
    "unit_tests": [
        "Run focused failing tests first and fix regressions before retrying validate_release.",
        "Inspect latest logs under reports/validation/logs/unit_tests.log.",
    ],
    "docker_compose_config": [
        "Validate docker compose configuration and required env vars.",
        "Run: docker compose config -q",
    ],
    "docker_compose_up_db": [
        "Restart database service and verify container health.",
        "Run: docker compose up -d db && docker compose ps",
    ],
    "db_connectivity": [
        "Verify DATABASE_URL and local DB availability.",
        "Run: python3 -m src.database create",
    ],
    "db_schema_create": [
        "Apply schema setup and re-check DB permissions.",
        "Run: python3 -m src.database create",
    ],
    "canonicalization_smoke": [
        "Fix datasource mapping and canonical contract issues.",
        "Run: python3 -m src.etl --dry-run --contract-profile canonical",
    ],
    "etl_smoke": [
        "Run ETL in dry-run mode and inspect contract failures.",
        "Review latest ETL report under reports/data/etl_run_*.md.",
    ],
    "model_smoke": [
        "Regenerate smoke model artifacts and verify feature availability.",
        "Run: python3 -m src.model --no-mlflow --optuna-trials 1",
    ],
    "evaluate_smoke": [
        "Recompute evaluation metrics from latest prediction artifact.",
        "Run: python3 -m src.evaluate --predictions-csv <path>",
    ],
    "explain_smoke": [
        "Rebuild SHAP artifacts and confirm model/explainability dependencies.",
        "Run: python3 -m src.explain --model-path <path> --evaluation-csv <path>",
    ],
    "artifact_inventory": [
        "Regenerate missing model/monitoring artifacts expected by smoke validation.",
        "Verify required files listed in validate_release artifact_inventory check.",
    ],
    "streamlit_app_smoke": [
        "Start app locally and inspect startup logs.",
        "Run: python3 -m streamlit run app.py --server.headless true",
    ],
    "mlflow_track_smoke": [
        "Check MLflow tracking URI and rerun smoke tracking step.",
        "Run: python3 -m src.mlops.track_run ...",
    ],
    "drift_monitor_smoke": [
        "Refresh reference/current slices and rerun drift monitor.",
        "Run: python3 -m src.monitoring.drift ...",
    ],
    "performance_monitor_smoke": [
        "Rebuild predictions artifact and rerun performance monitor.",
        "Run: python3 -m src.monitoring.performance ...",
    ],
    "retrain_policy_smoke": [
        "Recompute retrain decision inputs and rerun retrain policy.",
        "Run: python3 -m src.retrain_policy ...",
    ],
    "production_data_evidence": [
        "Generate production ETL markdown/csv evidence (non-smoke, non-dryrun tags).",
        "Ensure reports/data/etl_run_*.md and *.csv exist for production evidence.",
    ],
    "production_model_evidence": [
        "Regenerate production model v1/prod artifacts and metrics metadata.",
        "Ensure train_rows threshold and artifact_tag/model_version policy constraints pass.",
    ],
    "production_product_evidence": [
        "Ensure required product artifacts (including app.py) are present and valid.",
    ],
    "streamlit_app_production": [
        "Run Streamlit startup check in production mode and review logs.",
        "Run: python3 -m streamlit run app.py --server.headless true",
    ],
    "production_ops_evidence": [
        "Refresh monitoring and retrain policy artifacts for production evidence.",
        "Ensure reports/monitoring and reports/releases latest files are present.",
    ],
    "arena_governance_production": [
        "Generate/approve arena proposal before production validation.",
        "Run: ./scripts/ds_workflow.sh arena-propose",
    ],
    "release_tag": [
        "Confirm all core gates are green, then rerun with --tag-release if desired.",
    ],
}


def _failed_checks(report_payload: dict[str, Any]) -> dict[str, list[str]]:
    gates = report_payload.get("gates", {})
    failed: dict[str, list[str]] = {}
    for gate_name, gate in gates.items():
        gate_failed = list(gate.get("failed_checks", [])) + list(gate.get("missing_checks", []))
        if gate_failed:
            failed[gate_name] = gate_failed
    return failed


def _gate_sort_key(gate_name: str) -> tuple[int, str]:
    return (GATE_PRIORITY.get(gate_name, 999), gate_name)


def _gate_category(gate_name: str) -> str:
    if gate_name in GATE_CATEGORY_MAP:
        return GATE_CATEGORY_MAP[gate_name]
    if "(" in gate_name and ")" in gate_name:
        inner = gate_name[gate_name.find("(") + 1 : gate_name.find(")")].strip()
        if inner:
            return inner
    return "Unknown"


def _recommended_actions_for_check(check_name: str) -> list[str]:
    actions = CHECK_REPAIR_ACTIONS.get(check_name)
    if actions:
        return actions
    return [
        "Inspect the check log under reports/validation/logs and rerun the targeted command.",
        "Apply a focused fix, then rerun validate_release in smoke mode.",
    ]


def _gate_diagnosis(failed_checks: dict[str, list[str]]) -> list[dict[str, Any]]:
    diagnosis: list[dict[str, Any]] = []
    for gate_name in sorted(failed_checks, key=_gate_sort_key):
        checks = failed_checks[gate_name]
        diagnosis.append(
            {
                "gate": gate_name,
                "category": _gate_category(gate_name),
                "failed_checks": [
                    {
                        "name": check_name,
                        "recommended_actions": _recommended_actions_for_check(check_name),
                    }
                    for check_name in checks
                ],
            }
        )
    return diagnosis


def _load_repair_command_pack(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("repair command pack must be a JSON object")
    normalized: dict[str, list[str]] = {}
    for key, raw_value in payload.items():
        commands: list[str]
        if isinstance(raw_value, str):
            commands = [raw_value]
        elif isinstance(raw_value, list) and all(isinstance(item, str) for item in raw_value):
            commands = list(raw_value)
        else:
            raise ValueError(f"invalid command list for key '{key}'")
        cleaned = [cmd.strip() for cmd in commands if cmd.strip()]
        if cleaned:
            normalized[str(key)] = cleaned
    return normalized


def _resolve_pack_commands(
    failed_checks: dict[str, list[str]],
    repair_pack: dict[str, list[str]],
) -> list[str]:
    if not repair_pack:
        return []
    resolved: list[str] = []
    for gate_name in sorted(failed_checks, key=_gate_sort_key):
        gate_commands = repair_pack.get(gate_name, [])
        category_commands = repair_pack.get(_gate_category(gate_name), [])
        for command in [*gate_commands, *category_commands]:
            if command not in resolved:
                resolved.append(command)
    return resolved


def _build_remediation_brief(
    *,
    attempt: int,
    max_iterations: int,
    validate_command: list[str],
    failed_checks: dict[str, list[str]],
    repair_command_source: str,
    repair_command_candidates: list[str],
    repair_command_pack_path: str | None,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "max_iterations": max_iterations,
        "validate_command": " ".join(validate_command),
        "failed_checks": failed_checks,
        "gate_diagnosis": _gate_diagnosis(failed_checks),
        "repair_command_source": repair_command_source,
        "repair_command_candidates": repair_command_candidates,
        "repair_command_pack_path": repair_command_pack_path,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _build_validate_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python3",
        "-m",
        "src.validate_release",
        "--mode",
        args.mode,
        "--contract-profile",
        args.contract_profile,
        "--output-json",
        str(args.report_json),
        "--output-md",
        str(args.report_md),
    ]
    if args.mapping_yaml:
        cmd.extend(["--mapping-yaml", str(args.mapping_yaml)])
    if args.data_source:
        cmd.extend(["--data-source", args.data_source])
    return cmd


def run_loop(args: argparse.Namespace) -> int:
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_brief.parent.mkdir(parents=True, exist_ok=True)

    repair_command_pack = getattr(args, "repair_command_pack", None)
    try:
        repair_pack = _load_repair_command_pack(repair_command_pack)
    except Exception as exc:
        print(f"[autonomy] failed to load repair command pack: {exc}")
        return 2

    for attempt in range(1, args.max_iterations + 1):
        print(f"[autonomy] attempt {attempt}/{args.max_iterations}: running validate_release")
        cmd = _build_validate_command(args)
        result = subprocess.run(cmd, text=True)

        payload = {}
        if args.report_json.exists():
            try:
                payload = json.loads(args.report_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}

        failed = _failed_checks(payload)
        if result.returncode == 0 and not failed:
            print("[autonomy] all gates green")
            return 0

        pack_commands = _resolve_pack_commands(failed, repair_pack)
        if args.repair_command:
            repair_source = "explicit"
            repair_candidates = [args.repair_command]
        elif pack_commands:
            repair_source = "pack"
            repair_candidates = pack_commands
        else:
            repair_source = "none"
            repair_candidates = []

        brief = _build_remediation_brief(
            attempt=attempt,
            max_iterations=args.max_iterations,
            validate_command=cmd,
            failed_checks=failed,
            repair_command_source=repair_source,
            repair_command_candidates=repair_candidates,
            repair_command_pack_path=str(repair_command_pack) if repair_command_pack else None,
        )
        args.runtime_brief.write_text(json.dumps(brief, indent=2), encoding="utf-8")
        print(f"[autonomy] gates blocked; wrote remediation brief: {args.runtime_brief}")

        if args.repair_command:
            print(f"[autonomy] executing repair command: {args.repair_command}")
            repair = subprocess.run(args.repair_command, shell=True, text=True)
            if repair.returncode != 0:
                print("[autonomy] repair command failed; stopping loop")
                return repair.returncode
        elif pack_commands:
            for pack_command in pack_commands:
                print(f"[autonomy] executing repair command pack entry: {pack_command}")
                repair = subprocess.run(pack_command, shell=True, text=True)
                if repair.returncode != 0:
                    print("[autonomy] repair command failed; stopping loop")
                    return repair.returncode
        else:
            print("[autonomy] no repair command provided; stopping after first blocked attempt")
            return 1

    print("[autonomy] max iterations reached with blocked gates")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous validation loop runner for SPEC-NYC.")
    parser.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--contract-profile", choices=("nyc", "canonical"), default="canonical")
    parser.add_argument("--data-source", choices=("csv", "nyc_open_data", "postgres"), default="csv")
    parser.add_argument("--mapping-yaml", type=Path, default=Path("src/datasources/mappings/spec_nyc_v1.yaml"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/validation/autonomy_loop_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/validation/autonomy_loop_report.md"))
    parser.add_argument("--runtime-brief", type=Path, default=Path(".codex/runtime/repair_brief.json"))
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--repair-command", type=str, default=None, help="Optional shell command executed between failed attempts.")
    parser.add_argument(
        "--repair-command-pack",
        type=Path,
        default=None,
        help="Optional JSON command pack keyed by gate name (e.g. 'Gate A (Data)') or category (e.g. 'Data').",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    raise SystemExit(run_loop(parser.parse_args()))
