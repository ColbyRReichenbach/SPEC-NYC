import argparse
import json
from pathlib import Path

from src.autonomy.loop_runner import (
    _build_remediation_brief,
    _failed_checks,
    _gate_diagnosis,
    _resolve_pack_commands,
    run_loop,
)


def test_failed_checks_extracts_failed_and_missing():
    payload = {
        "gates": {
            "Gate A (Data)": {"failed_checks": ["etl_smoke"], "missing_checks": []},
            "Gate B (Model)": {"failed_checks": [], "missing_checks": ["model_smoke"]},
            "Gate C (Product)": {"failed_checks": [], "missing_checks": []},
        }
    }
    failed = _failed_checks(payload)
    assert failed["Gate A (Data)"] == ["etl_smoke"]
    assert failed["Gate B (Model)"] == ["model_smoke"]
    assert "Gate C (Product)" not in failed


def test_gate_diagnosis_includes_category_and_recommended_actions():
    diagnosis = _gate_diagnosis(
        {
            "Gate B (Model)": ["model_smoke"],
            "Gate A (Data)": ["etl_smoke"],
        }
    )
    assert diagnosis[0]["gate"] == "Gate A (Data)"
    assert diagnosis[0]["category"] == "Data"
    assert diagnosis[0]["failed_checks"][0]["name"] == "etl_smoke"
    assert diagnosis[0]["failed_checks"][0]["recommended_actions"]


def test_resolve_pack_commands_supports_gate_and_category_keys():
    failed = {
        "Gate A (Data)": ["etl_smoke"],
        "Gate C (Product)": ["streamlit_app_smoke"],
    }
    pack = {
        "Gate A (Data)": ["echo repair-gate-a"],
        "Product": ["echo repair-product"],
    }
    commands = _resolve_pack_commands(failed, pack)
    assert commands == ["echo repair-gate-a", "echo repair-product"]


def test_build_remediation_brief_keeps_machine_readable_fields():
    brief = _build_remediation_brief(
        attempt=1,
        max_iterations=3,
        validate_command=["python3", "-m", "src.validate_release"],
        failed_checks={"Gate A (Data)": ["etl_smoke"]},
        repair_command_source="pack",
        repair_command_candidates=["echo repair-data"],
        repair_command_pack_path="tmp/pack.json",
    )
    assert brief["failed_checks"]["Gate A (Data)"] == ["etl_smoke"]
    assert brief["gate_diagnosis"][0]["category"] == "Data"
    assert brief["repair_command_source"] == "pack"
    assert brief["repair_command_candidates"] == ["echo repair-data"]


def test_run_loop_executes_pack_repairs_and_retries(tmp_path, monkeypatch):
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    runtime_brief = tmp_path / "repair_brief.json"
    repair_pack = tmp_path / "repair_pack.json"
    repair_pack.write_text(json.dumps({"Data": ["echo repair-data"]}), encoding="utf-8")

    calls: list[tuple[str, object]] = []
    state = {"validate_calls": 0}

    class Result:
        def __init__(self, returncode: int):
            self.returncode = returncode

    def fake_run(command, *_, **kwargs):
        shell = bool(kwargs.get("shell", False))
        calls.append(("shell" if shell else "cmd", command))
        if shell:
            return Result(0)
        if isinstance(command, list) and command[:3] == ["python3", "-m", "src.validate_release"]:
            state["validate_calls"] += 1
            if state["validate_calls"] == 1:
                report_json.write_text(
                    json.dumps(
                        {
                            "gates": {
                                "Gate A (Data)": {"failed_checks": ["etl_smoke"], "missing_checks": []},
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return Result(1)
            report_json.write_text(json.dumps({"gates": {}}, indent=2), encoding="utf-8")
            return Result(0)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("src.autonomy.loop_runner.subprocess.run", fake_run)

    args = argparse.Namespace(
        mode="smoke",
        contract_profile="canonical",
        data_source="csv",
        mapping_yaml=Path("src/datasources/mappings/spec_nyc_v1.yaml"),
        report_json=report_json,
        report_md=report_md,
        runtime_brief=runtime_brief,
        max_iterations=3,
        repair_command=None,
        repair_command_pack=repair_pack,
    )

    rc = run_loop(args)
    assert rc == 0
    assert state["validate_calls"] == 2
    assert ("shell", "echo repair-data") in calls

    brief = json.loads(runtime_brief.read_text(encoding="utf-8"))
    assert brief["repair_command_source"] == "pack"
    assert brief["repair_command_candidates"] == ["echo repair-data"]
    assert brief["gate_diagnosis"][0]["gate"] == "Gate A (Data)"
