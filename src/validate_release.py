"""W6 release validator for S.P.E.C. NYC."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DB_URL = "postgresql://spec:spec_password@localhost:5433/spec_nyc"
REPORT_DIR = Path("reports/validation")
LOG_DIR = REPORT_DIR / "logs"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    command: str | None = None
    duration_sec: float = 0.0
    log_path: str | None = None
    artifacts: List[str] | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def _slug(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _write_log(name: str, content: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{_slug(name)}.log"
    path.write_text(content, encoding="utf-8")
    return path


def run_command(
    name: str,
    command: str,
    *,
    timeout_sec: int = 300,
    env_overrides: Dict[str, str] | None = None,
) -> CheckResult:
    started = time.time()
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            env=env,
        )
        duration = time.time() - started
        status = "pass" if proc.returncode == 0 else "fail"
        tail_lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-40:]
        detail = (
            f"exit={proc.returncode}; "
            + (tail_lines[-1] if tail_lines else "no output")
        )
        log_path = _write_log(
            name,
            "\n".join(
                [
                    f"$ {command}",
                    "",
                    "STDOUT:",
                    proc.stdout,
                    "",
                    "STDERR:",
                    proc.stderr,
                ]
            ),
        )
        return CheckResult(
            name=name,
            status=status,
            detail=detail,
            command=command,
            duration_sec=duration,
            log_path=str(log_path),
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - started
        log_path = _write_log(
            name,
            "\n".join(
                [
                    f"$ {command}",
                    "",
                    f"TIMEOUT after {timeout_sec}s",
                    "",
                    "PARTIAL STDOUT:",
                    (exc.stdout or ""),
                    "",
                    "PARTIAL STDERR:",
                    (exc.stderr or ""),
                ]
            ),
        )
        return CheckResult(
            name=name,
            status="fail",
            detail=f"timeout after {timeout_sec}s",
            command=command,
            duration_sec=duration,
            log_path=str(log_path),
        )


def _build_etl_smoke_input(path: Path, rows: int = 260, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    category_pairs = [
        ("01", "A1"),
        ("02", "B2"),
        ("03", "C0"),
        ("08", "D4"),
        ("10", "D7"),
        ("12", "R2"),
        ("13", "R4"),
        ("14", "S2"),
        ("15", "R8"),
        ("17", "R9"),
    ]
    neighborhoods = ["Chelsea", "Harlem", "Bushwick", "Astoria", "Park Slope", "LIC"]
    rows_out = []
    for i in range(rows):
        borough = int(rng.choice([1, 3, 4, 5]))
        block = int(10000 + (i % 80000))
        lot = int(10 + (i % 9000))
        bbl = int(f"{borough}{block:05d}{lot:04d}")
        prefix, bclass = category_pairs[i % len(category_pairs)]
        lat = float(40.55 + rng.random() * 0.35)
        lon = float(-74.15 + rng.random() * 0.45)
        sale_price = int(200_000 + rng.normal(0, 60_000) + (i % 8) * 85_000)
        sale_price = max(sale_price, 25_000)
        sale_date = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(i % 390))).date().isoformat()
        gross_sqft: float | None = float(500 + (i % 7) * 250 + max(rng.normal(0, 40), -200))
        if i % 23 == 0:
            gross_sqft = 0.0
        elif i % 31 == 0:
            gross_sqft = np.nan
        year_built = int(1920 + (i % 95))
        if i % 19 == 0:
            year_built = 0
        apt = ""
        if prefix in {"08", "10", "12", "13", "15", "17"}:
            apt = f"{1 + (i % 25)}A"
        rows_out.append(
            {
                "sale_date": sale_date,
                "sale_price": sale_price,
                "bbl": bbl,
                "borough": borough,
                "block": block,
                "lot": lot,
                "latitude": lat,
                "longitude": lon,
                "neighborhood": neighborhoods[i % len(neighborhoods)],
                "building_class_category": f"{prefix} RESIDENTIAL",
                "building_class_at_time_of": bclass,
                "apartment_number": apt,
                "gross_square_feet": gross_sqft,
                "year_built": year_built,
                "residential_units": float(1 + (i % 8)),
                "total_units": float(1 + (i % 12)),
                "land_square_feet": float(900 + (i % 6) * 220),
            }
        )
    frame = pd.DataFrame(rows_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _build_training_smoke_input(path: Path, rows: int = 800, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    segment_to_classes = {
        "SINGLE_FAMILY": ["A1", "A5", "B2"],
        "ELEVATOR": ["D4", "D7", "R4"],
        "WALKUP": ["C6", "C7", "R2"],
        "SMALL_MULTI": ["S2", "S4", "R8"],
    }
    tiers = ["entry", "core", "premium", "luxury"]
    neighborhoods = ["Chelsea", "Harlem", "Bushwick", "Astoria", "Greenpoint", "Sunset Park"]
    segments = list(segment_to_classes.keys())
    borough_for_segment = {"SINGLE_FAMILY": "3", "ELEVATOR": "1", "WALKUP": "3", "SMALL_MULTI": "4"}
    segment_base = {"SINGLE_FAMILY": 120_000, "ELEVATOR": 450_000, "WALKUP": 260_000, "SMALL_MULTI": 320_000}

    rows_out = []
    for i in range(rows):
        segment = segments[i % len(segments)]
        tier = tiers[(i // len(segments)) % len(tiers)]
        sqft = float(450 + (i % 14) * 180 + rng.normal(0, 25))
        year_built = int(1910 + (i % 105))
        building_age = max(0, 2026 - year_built)
        distance = float(abs(rng.normal(6, 2.2)))
        residential_units = float(1 + (i % 10))
        total_units = float(residential_units + (i % 3))
        class_code = segment_to_classes[segment][i % len(segment_to_classes[segment])]
        borough = borough_for_segment[segment]
        neighborhood = neighborhoods[i % len(neighborhoods)]
        h3_idx = f"h3_{(i % 90):03d}"
        tier_bump = {"entry": 0, "core": 80_000, "premium": 180_000, "luxury": 380_000}[tier]
        noise = rng.normal(0, 45_000)
        sale_price = max(
            20_000,
            int(segment_base[segment] + tier_bump + sqft * 620 + (12 - distance) * 18_000 + noise),
        )
        sale_date = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i % 720))).date().isoformat()
        rows_out.append(
            {
                "sale_date": sale_date,
                "sale_price": sale_price,
                "h3_index": h3_idx,
                "gross_square_feet": sqft,
                "year_built": year_built,
                "building_age": building_age,
                "residential_units": residential_units,
                "total_units": total_units,
                "distance_to_center_km": distance,
                "borough": borough,
                "building_class": class_code,
                "property_segment": segment,
                "price_tier": tier,
                "neighborhood": neighborhood,
            }
        )
    frame = pd.DataFrame(rows_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def run_streamlit_smoke(port: int = 8504, timeout_sec: int = 40) -> CheckResult:
    name = "streamlit_app_smoke"
    command = (
        f"python3 -m streamlit run app.py --server.headless true "
        f"--server.address 127.0.0.1 --server.port {port}"
    )
    started = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    time.sleep(min(timeout_sec, 12))
    success = proc.poll() is None
    if success:
        proc.terminate()
        try:
            stdout, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate(timeout=5)
    else:
        stdout, _ = proc.communicate(timeout=5)
    duration = time.time() - started
    log_path = _write_log(name, f"$ {command}\n\nOUTPUT:\n{stdout}")
    return CheckResult(
        name=name,
        status="pass" if success else "fail",
        detail="app process stayed healthy for startup window" if success else "streamlit exited before startup window",
        command=command,
        duration_sec=duration,
        log_path=str(log_path),
    )


def check_artifacts(required_paths: Iterable[Path], pattern_paths: Iterable[str]) -> CheckResult:
    missing = [str(path) for path in required_paths if not path.exists()]
    matched = []
    missing_patterns = []
    for pattern in pattern_paths:
        files = [Path(p) for p in sorted(glob.glob(pattern))]
        if files:
            matched.extend(str(p) for p in files)
        else:
            missing_patterns.append(pattern)

    status = "pass" if not missing and not missing_patterns else "fail"
    detail_parts = []
    if missing:
        detail_parts.append(f"missing files: {', '.join(missing)}")
    if missing_patterns:
        detail_parts.append(f"no matches for: {', '.join(missing_patterns)}")
    if not detail_parts:
        detail_parts.append("all required artifacts present")
    return CheckResult(
        name="artifact_inventory",
        status=status,
        detail="; ".join(detail_parts),
        artifacts=matched + [str(p) for p in required_paths if p.exists()],
    )


def evaluate_gates(checks: List[CheckResult]) -> Dict[str, Dict[str, object]]:
    by_name = {check.name: check for check in checks}

    gate_map = {
        "Gate A (Data)": [
            "unit_tests",
            "docker_compose_config",
            "docker_compose_up_db",
            "db_connectivity",
            "db_schema_create",
            "etl_smoke",
        ],
        "Gate B (Model)": ["model_smoke", "evaluate_smoke", "explain_smoke", "artifact_inventory"],
        "Gate C (Product)": ["streamlit_app_smoke"],
        "Gate D (Ops)": ["mlflow_track_smoke", "drift_monitor_smoke", "performance_monitor_smoke", "retrain_policy_smoke"],
    }
    gate_status: Dict[str, Dict[str, object]] = {}
    for gate_name, required in gate_map.items():
        missing = [name for name in required if name not in by_name]
        failed = [name for name in required if name in by_name and not by_name[name].passed]
        passed = not missing and not failed
        gate_status[gate_name] = {
            "status": "done" if passed else "blocked",
            "required_checks": required,
            "failed_checks": failed,
            "missing_checks": missing,
        }

    all_core_green = all(entry["status"] == "done" for entry in gate_status.values())
    release_missing = []
    release_failed = []
    release_check = by_name.get("release_tag")
    if release_check is None:
        release_missing.append("release_tag")
    elif not release_check.passed:
        release_failed.append("release_tag")
    all_green = all_core_green and not release_missing and not release_failed
    gate_status["Gate E (Release)"] = {
        "status": "done" if all_green else "blocked",
        "required_checks": list(gate_status.keys()) + ["release_tag"],
        "failed_checks": release_failed,
        "missing_checks": release_missing,
        "all_green": all_green,
    }
    return gate_status


def maybe_tag_release(enabled: bool) -> CheckResult:
    if not enabled:
        return CheckResult(name="release_tag", status="pass", detail="tagging skipped (--tag-release not set)")

    has_tag = run_command("release_tag_check_existing", "git tag --list v1.0")
    if has_tag.passed:
        tag_text = Path(has_tag.log_path).read_text(encoding="utf-8")
        if "\nv1.0\n" in f"\n{tag_text}\n":
            return CheckResult(name="release_tag", status="pass", detail="v1.0 already exists")

    cmd = "git tag -a v1.0 -m \"S.P.E.C. NYC v1.0\""
    tag_result = run_command("release_tag_create", cmd)
    if not tag_result.passed:
        return CheckResult(
            name="release_tag",
            status="fail",
            detail=f"failed to create v1.0 tag ({tag_result.detail})",
            command=cmd,
            log_path=tag_result.log_path,
        )
    return CheckResult(name="release_tag", status="pass", detail="created git tag v1.0", command=cmd, log_path=tag_result.log_path)


def write_report(
    *,
    checks: List[CheckResult],
    gates: Dict[str, Dict[str, object]],
    output_md: Path,
    output_json: Path,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_sec": round((finished_at - started_at).total_seconds(), 2),
        "checks": [asdict(check) for check in checks],
        "gates": gates,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S.P.E.C. NYC V1 Readiness Report",
        "",
        f"- Started (UTC): `{started_at.isoformat()}`",
        f"- Finished (UTC): `{finished_at.isoformat()}`",
        f"- Duration: `{payload['duration_sec']}s`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail | Log |",
        "|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.name} | {check.status} | {check.detail.replace('|', '/')} | "
            f"{check.log_path or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Status | Failed Checks | Missing Checks |",
            "|---|---|---|---|",
        ]
    )
    for gate_name, gate in gates.items():
        failed = ", ".join(gate.get("failed_checks", [])) or "-"
        missing = ", ".join(gate.get("missing_checks", [])) or "-"
        lines.append(f"| {gate_name} | {gate.get('status')} | {failed} | {missing} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON payload: `{output_json}`",
            f"- Command logs: `{LOG_DIR}`",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def _prepare_monitoring_inputs(smoke_dir: Path, training_csv: Path) -> Tuple[Path, Path]:
    reference_path = Path("reports/monitoring/reference_slice_v1.csv")
    current_path = Path("reports/monitoring/current_slice_v1.csv")
    if reference_path.exists() and current_path.exists():
        return reference_path, current_path

    frame = pd.read_csv(training_csv)
    cutoff = int(len(frame) * 0.7)
    reference_fallback = smoke_dir / "reference_slice_smoke.csv"
    current_fallback = smoke_dir / "current_slice_smoke.csv"
    frame.iloc[:cutoff].to_csv(reference_fallback, index=False)
    frame.iloc[cutoff:].to_csv(current_fallback, index=False)
    return reference_fallback, current_fallback


def run_validation(args: argparse.Namespace) -> int:
    started_at = datetime.utcnow()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    smoke_dir = REPORT_DIR / "smoke_inputs"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    db_env = {"DATABASE_URL": args.database_url}

    etl_input = _build_etl_smoke_input(smoke_dir / "etl_raw_smoke.csv")
    training_input = _build_training_smoke_input(smoke_dir / "train_smoke.csv")
    monitoring_reference, monitoring_current = _prepare_monitoring_inputs(smoke_dir, training_input)

    checks: List[CheckResult] = []

    checks.append(run_command("unit_tests", "python3 -m unittest tests.test_etl tests.test_evaluate tests.test_ai_security tests.test_monitoring -v", timeout_sec=300))
    checks.append(run_command("docker_compose_config", "docker compose config -q", timeout_sec=60))
    checks.append(run_command("docker_compose_up_db", "docker compose up -d db", timeout_sec=180))
    checks.append(
        run_command(
            "db_connectivity",
            (
                "python3 -c \"from sqlalchemy import create_engine,text; "
                "import os; "
                "e=create_engine(os.environ['DATABASE_URL']); "
                "conn=e.connect(); conn.execute(text('SELECT 1')); conn.close(); print('db_ok')\""
            ),
            timeout_sec=60,
            env_overrides=db_env,
        )
    )
    checks.append(run_command("db_schema_create", "python3 -m src.database create", timeout_sec=120, env_overrides=db_env))
    checks.append(
        run_command(
            "etl_smoke",
            f"python3 -m src.etl --input {etl_input} --limit 220 --dry-run --write-report",
            timeout_sec=240,
        )
    )
    checks.append(
        run_command(
            "model_smoke",
            f"python3 -m src.model --input-csv {training_input} --model-version {args.smoke_model_version} "
            "--optuna-trials 1 --shap-sample-size 120 --no-mlflow",
            timeout_sec=600,
        )
    )
    checks.append(
        run_command(
            "evaluate_smoke",
            (
                "python3 -m src.evaluate "
                f"--predictions-csv reports/model/evaluation_predictions_{args.smoke_model_version}.csv "
                f"--output-json models/metrics_{args.smoke_model_version}_eval.json "
                f"--segment-scorecard-csv reports/model/segment_scorecard_{args.smoke_model_version}_eval.csv"
            ),
            timeout_sec=120,
        )
    )
    checks.append(
        run_command(
            "explain_smoke",
            (
                "python3 -m src.explain "
                f"--model-path models/model_{args.smoke_model_version}.joblib "
                f"--evaluation-csv reports/model/evaluation_predictions_{args.smoke_model_version}.csv "
                f"--summary-plot-path reports/model/shap_summary_{args.smoke_model_version}.png "
                f"--waterfall-plot-path reports/model/shap_waterfall_{args.smoke_model_version}.png "
                "--sample-size 120"
            ),
            timeout_sec=240,
        )
    )
    checks.append(
        run_command(
            "mlflow_track_smoke",
            (
                "python3 -m src.mlops.track_run "
                f"--metrics-json models/metrics_{args.smoke_model_version}.json "
                f"--model-artifact models/model_{args.smoke_model_version}.joblib "
                f"--scorecard-csv reports/model/segment_scorecard_{args.smoke_model_version}.csv "
                f"--predictions-csv reports/model/evaluation_predictions_{args.smoke_model_version}.csv "
                "--run-name w6-validate-smoke "
                "--dataset-version w6_smoke"
            ),
            timeout_sec=180,
        )
    )
    checks.append(
        run_command(
            "drift_monitor_smoke",
            (
                "python3 -m src.monitoring.drift "
                f"--reference-csv {monitoring_reference} "
                f"--current-csv {monitoring_current} "
                "--output-csv reports/monitoring/drift_w6_smoke.csv "
                "--output-json reports/monitoring/drift_w6_smoke.json "
                "--output-md reports/monitoring/drift_w6_smoke.md"
            ),
            timeout_sec=120,
        )
    )
    checks.append(
        run_command(
            "performance_monitor_smoke",
            (
                "python3 -m src.monitoring.performance "
                f"--predictions-csv reports/model/evaluation_predictions_{args.smoke_model_version}.csv "
                "--output-json reports/monitoring/performance_w6_smoke.json "
                "--output-md reports/monitoring/performance_w6_smoke.md"
            ),
            timeout_sec=120,
        )
    )
    checks.append(
        run_command(
            "retrain_policy_smoke",
            (
                "python3 -m src.retrain_policy "
                f"--metrics-json models/metrics_{args.smoke_model_version}.json "
                "--performance-json reports/monitoring/performance_w6_smoke.json "
                "--drift-csv reports/monitoring/drift_w6_smoke.csv "
                "--output-json reports/releases/retrain_decision_w6_smoke.json "
                "--output-md reports/releases/retrain_decision_w6_smoke.md"
            ),
            timeout_sec=120,
        )
    )
    checks.append(run_streamlit_smoke(port=args.streamlit_port, timeout_sec=40))

    checks.append(
        check_artifacts(
            required_paths=[
                Path("models/model_v1.joblib"),
                Path("models/metrics_v1.json"),
                Path("reports/model/segment_scorecard_v1.csv"),
                Path("reports/model/shap_summary_v1.png"),
                Path("reports/model/shap_waterfall_v1.png"),
                Path("reports/monitoring/drift_latest.json"),
                Path("reports/monitoring/performance_latest.json"),
                Path("reports/releases/retrain_decision_latest.json"),
                Path("app.py"),
            ],
            pattern_paths=["reports/data/etl_run_*.md", "reports/data/etl_run_*.csv"],
        )
    )

    checks.append(run_command("docker_compose_stop_db", "docker compose stop db", timeout_sec=120))

    pre_tag_gates = evaluate_gates(checks)
    core_gates_green = all(
        pre_tag_gates.get(gate, {}).get("status") == "done"
        for gate in ("Gate A (Data)", "Gate B (Model)", "Gate C (Product)", "Gate D (Ops)")
    )
    tag_result = maybe_tag_release(enabled=args.tag_release and core_gates_green)
    checks.append(tag_result)
    gates = evaluate_gates(checks)
    gate_e_green = bool(gates.get("Gate E (Release)", {}).get("all_green"))

    finished_at = datetime.utcnow()
    write_report(
        checks=checks,
        gates=gates,
        output_md=args.output_md,
        output_json=args.output_json,
        started_at=started_at,
        finished_at=finished_at,
    )

    print(json.dumps({"gate_e_all_green": gate_e_green, "report_md": str(args.output_md), "report_json": str(args.output_json)}, indent=2))
    return 0 if gate_e_green else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate S.P.E.C. NYC v1 release readiness (W6).")
    parser.add_argument("--output-md", type=Path, default=Path("reports/validation/v1_readiness_report.md"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/validation/v1_readiness_report.json"))
    parser.add_argument("--database-url", type=str, default=DEFAULT_DB_URL)
    parser.add_argument("--smoke-model-version", type=str, default="v1_smoke")
    parser.add_argument("--streamlit-port", type=int, default=8504)
    parser.add_argument("--tag-release", action="store_true", help="Create git tag v1.0 when all gates are green.")
    return parser


def _cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run_validation(args))


if __name__ == "__main__":
    _cli()
