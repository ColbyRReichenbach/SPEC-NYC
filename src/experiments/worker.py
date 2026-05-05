"""Run governed experiment training jobs from dashboard-created manifests.

The dashboard queues immutable experiment specs. This worker is the execution
boundary: it reads a queued job_manifest.json, runs the repository trainer, and
writes the audit artifacts needed to compare the challenger against the locked
champion dataset contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_experiment_job(repo_root: Path, experiment_id: str) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    paths = _experiment_paths(repo_root, experiment_id)
    bundle = _read_json(paths["summary"])
    job = _read_json(paths["job_manifest"])

    if bundle.get("status") != "queued" or job.get("status") != "queued":
        raise RuntimeError(
            f"Experiment {experiment_id} is not queued "
            f"(summary={bundle.get('status')}, job={job.get('status')})."
        )

    _mark_job_running(repo_root, paths, bundle, job)
    before_packages = _list_model_packages(repo_root)

    stdout_path = repo_root / job["logs"]["stdout"]
    stderr_path = repo_root / job["logs"]["stderr"]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    return_code = 1
    failure: str | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            stdout.write(f"experiment_id={experiment_id}\n")
            stdout.write(f"job_id={job['id']}\n")
            stdout.write(f"started_at={job['started_at']}\n\n")
            stdout.flush()
            result = subprocess.run(job["command"], cwd=repo_root, stdout=stdout, stderr=stderr, check=False)
            return_code = int(result.returncode)
    except Exception as exc:  # pragma: no cover - defensive failure path
        failure = str(exc)

    after_packages = _list_model_packages(repo_root)
    created_packages = [package for package in after_packages if package not in before_packages]

    if return_code != 0 or failure:
        _mark_job_failed(repo_root, paths, bundle, job, return_code, failure or "Training command failed.")
        return {"status": "failed", "experiment_id": experiment_id, "exit_code": return_code}

    package_path = _select_newest_package(repo_root, created_packages)
    if package_path is None:
        _mark_job_failed(
            repo_root,
            paths,
            bundle,
            job,
            return_code,
            "Training command succeeded but no new model package was detected.",
        )
        return {"status": "failed", "experiment_id": experiment_id, "exit_code": return_code}

    comparison = build_comparison_report(repo_root, bundle, package_path)
    _mark_job_completed(repo_root, paths, bundle, job, package_path, comparison)
    return {
        "status": "completed",
        "experiment_id": experiment_id,
        "model_package_path": _relative_path(repo_root, package_path),
        "comparison_status": comparison["status"],
    }


def build_comparison_report(repo_root: Path, bundle: dict[str, Any], package_path: Path) -> dict[str, Any]:
    metrics = _read_json(package_path / "metrics.json")
    data_manifest = _read_json(package_path / "data_manifest.json")
    metadata = metrics.get("metadata", {})
    overall = metrics.get("overall", {})
    candidate_metrics = {
        "ppe10": _number(overall.get("ppe10")),
        "mdape": _number(overall.get("mdape")),
        "r2": overall.get("r2") if isinstance(overall.get("r2"), (int, float)) else None,
    }
    champion_metrics = bundle["baseline_metrics"]
    candidate_split_signature = build_split_signature(
        dataset_version=str(metadata.get("dataset_version") or data_manifest.get("dataset_version") or "unknown"),
        data_snapshot_sha256=str(data_manifest.get("data_snapshot_sha256") or "unknown"),
        train_rows=int(metadata.get("train_rows") or 0),
        test_rows=int(metadata.get("test_rows") or 0),
        min_sale_date=str(data_manifest.get("min_sale_date") or "unknown"),
        max_sale_date=str(data_manifest.get("max_sale_date") or "unknown"),
    )
    locked_snapshot = bundle["dataset_snapshot"]
    same_dataset_contract = (
        str(data_manifest.get("data_snapshot_sha256") or "unknown") == locked_snapshot["data_snapshot_sha256"]
        and candidate_split_signature == locked_snapshot["split_signature_sha256"]
    )
    mdape_delta = candidate_metrics["mdape"] - float(champion_metrics["mdape"])
    ppe10_delta = candidate_metrics["ppe10"] - float(champion_metrics["ppe10"])
    r2_delta = (
        candidate_metrics["r2"] - float(champion_metrics["r2"])
        if candidate_metrics["r2"] is not None and champion_metrics.get("r2") is not None
        else None
    )
    metric_gate_passed = mdape_delta <= 0 and ppe10_delta >= -0.005

    if not same_dataset_contract:
        status = "failed"
        blocking_reason = "Challenger package did not match the locked dataset snapshot and split signature."
    elif metric_gate_passed:
        status = "passed"
        blocking_reason = "Comparison passed metric gates; promotion still requires manual champion approval."
    else:
        status = "failed"
        blocking_reason = "Comparison completed but challenger did not beat champion guardrails."

    return {
        "status": status,
        "required_before_promotion": True,
        "champion_package_id": bundle["run_plan"]["baseline_package_id"],
        "challenger_package_id": str(metadata.get("model_package_id") or package_path.name),
        "same_dataset_required": True,
        "same_dataset_contract": same_dataset_contract,
        "dataset_snapshot_sha256": locked_snapshot["data_snapshot_sha256"],
        "split_signature_sha256": locked_snapshot["split_signature_sha256"],
        "candidate_split_signature_sha256": candidate_split_signature,
        "champion_metrics": champion_metrics,
        "challenger_metrics": candidate_metrics,
        "metric_deltas": {
            "ppe10": ppe10_delta,
            "mdape": mdape_delta,
            "r2": r2_delta,
        },
        "blocking_reason": blocking_reason,
        "generated_at": _utc_now(),
    }


def build_split_signature(
    *,
    dataset_version: str,
    data_snapshot_sha256: str,
    train_rows: int,
    test_rows: int,
    min_sale_date: str,
    max_sale_date: str,
) -> str:
    raw = "|".join(
        [
            dataset_version,
            data_snapshot_sha256,
            str(train_rows),
            str(test_rows),
            min_sale_date,
            max_sale_date,
            "time_ordered_split",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mark_job_running(
    repo_root: Path,
    paths: dict[str, Path],
    bundle: dict[str, Any],
    job: dict[str, Any],
) -> None:
    now = _utc_now()
    job.update({"status": "running", "started_at": now, "exit_code": None})
    bundle.update({"status": "running", "updated_at": now, "training_job": job})
    _write_json(paths["job_manifest"], job)
    _write_summary_and_manifest(paths, bundle, job, "training_job_started")
    _append_audit(
        repo_root,
        bundle,
        {
            "event": "training_job_started",
            "created_at": now,
            "experiment_id": bundle["id"],
            "job_id": job["id"],
            "command_display": job["command_display"],
        },
    )


def _mark_job_failed(
    repo_root: Path,
    paths: dict[str, Path],
    bundle: dict[str, Any],
    job: dict[str, Any],
    return_code: int,
    error: str,
) -> None:
    now = _utc_now()
    job.update({"status": "failed", "failed_at": now, "exit_code": return_code, "error": error})
    bundle.update({"status": "failed", "updated_at": now, "training_job": job})
    bundle["comparison"].update({"status": "failed", "blocking_reason": error})
    _write_json(paths["job_manifest"], job)
    _write_json(paths["comparison_report"], bundle["comparison"])
    _write_summary_and_manifest(paths, bundle, job, "training_job_failed")
    _append_audit(
        repo_root,
        bundle,
        {
            "event": "training_job_failed",
            "created_at": now,
            "experiment_id": bundle["id"],
            "job_id": job["id"],
            "exit_code": return_code,
            "error": error,
        },
    )


def _mark_job_completed(
    repo_root: Path,
    paths: dict[str, Path],
    bundle: dict[str, Any],
    job: dict[str, Any],
    package_path: Path,
    comparison: dict[str, Any],
) -> None:
    now = _utc_now()
    package_rel = _relative_path(repo_root, package_path)
    package_id = comparison["challenger_package_id"]
    job.update(
        {
            "status": "completed",
            "completed_at": now,
            "exit_code": 0,
            "output": {
                "model_package_id": package_id,
                "model_package_path": package_rel,
            },
        }
    )
    bundle["run_plan"]["challenger_package_id"] = package_id
    bundle.update({"status": "completed", "updated_at": now, "training_job": job})
    bundle["comparison"] = {
        "status": comparison["status"],
        "champion_package_id": comparison["champion_package_id"],
        "challenger_package_id": package_id,
        "same_dataset_required": True,
        "dataset_snapshot_sha256": comparison["dataset_snapshot_sha256"],
        "split_signature_sha256": comparison["split_signature_sha256"],
        "blocking_reason": comparison["blocking_reason"],
    }
    _write_json(paths["job_manifest"], job)
    _write_json(paths["comparison_report"], comparison)
    _write_summary_and_manifest(paths, bundle, job, "training_job_completed")
    _append_audit(
        repo_root,
        bundle,
        {
            "event": "training_job_completed",
            "created_at": now,
            "experiment_id": bundle["id"],
            "job_id": job["id"],
            "model_package_id": package_id,
            "model_package_path": package_rel,
            "comparison_status": comparison["status"],
        },
    )


def _write_summary_and_manifest(
    paths: dict[str, Path],
    bundle: dict[str, Any],
    job: dict[str, Any],
    lifecycle_event: str,
) -> None:
    manifest = _read_json(paths["run_manifest"])
    lifecycle = list(dict.fromkeys([*manifest.get("lifecycle", []), lifecycle_event]))
    manifest.update(
        {
            "updated_at": bundle.get("updated_at"),
            "status": bundle["status"],
            "lifecycle": lifecycle,
            "training_job": job,
            "artifact_paths": bundle["artifact_paths"],
        }
    )
    _write_json(paths["summary"], bundle)
    _write_json(paths["run_manifest"], manifest)


def _experiment_paths(repo_root: Path, experiment_id: str) -> dict[str, Path]:
    if not experiment_id.startswith("exp_") or "/" in experiment_id or ".." in experiment_id:
        raise ValueError(f"Invalid experiment id: {experiment_id}")

    run_dir = repo_root / "reports" / "experiments" / "runs" / experiment_id
    return {
        "summary": repo_root / "reports" / "experiments" / f"{experiment_id}.json",
        "run_dir": run_dir,
        "run_manifest": run_dir / "run_manifest.json",
        "job_manifest": run_dir / "job_manifest.json",
        "comparison_report": run_dir / "comparison_report.json",
    }


def _list_model_packages(repo_root: Path) -> set[Path]:
    packages_dir = repo_root / "models" / "packages"
    if not packages_dir.exists():
        return set()
    return {path for path in packages_dir.iterdir() if path.is_dir() and path.name.startswith("spec_nyc_avm_")}


def _select_newest_package(repo_root: Path, packages: list[Path]) -> Path | None:
    if not packages:
        return None
    return max(packages, key=lambda path: path.stat().st_mtime if path.exists() else 0).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_audit(repo_root: Path, bundle: dict[str, Any], event: dict[str, Any]) -> None:
    audit_path = repo_root / bundle["artifact_paths"]["audit_log"]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _relative_path(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run one governed experiment training job.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--once", action="store_true", help="Run exactly one queued experiment job.")
    args = parser.parse_args()

    if not args.once:
        raise SystemExit("--once is required until the worker daemon mode is implemented.")

    result = run_experiment_job(args.repo_root, args.experiment_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
