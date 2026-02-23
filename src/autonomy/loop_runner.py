from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _failed_checks(report_payload: dict[str, Any]) -> dict[str, list[str]]:
    gates = report_payload.get("gates", {})
    failed: dict[str, list[str]] = {}
    for gate_name, gate in gates.items():
        gate_failed = list(gate.get("failed_checks", [])) + list(gate.get("missing_checks", []))
        if gate_failed:
            failed[gate_name] = gate_failed
    return failed


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

        brief = {
            "attempt": attempt,
            "max_iterations": args.max_iterations,
            "validate_command": " ".join(cmd),
            "failed_checks": failed,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        args.runtime_brief.write_text(json.dumps(brief, indent=2), encoding="utf-8")
        print(f"[autonomy] gates blocked; wrote remediation brief: {args.runtime_brief}")

        if args.repair_command:
            print(f"[autonomy] executing repair command: {args.repair_command}")
            repair = subprocess.run(args.repair_command, shell=True, text=True)
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
    return parser


if __name__ == "__main__":
    parser = build_parser()
    raise SystemExit(run_loop(parser.parse_args()))
