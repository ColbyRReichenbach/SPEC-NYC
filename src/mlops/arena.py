"""Champion/Challenger arena for S.P.E.C. NYC model lifecycle."""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ARENA_DIR = Path("reports/arena")
DEFAULT_POLICY_PATH = Path("config/arena_policy.yaml")
DEFAULT_EXPERIMENT_NAME = "spec-nyc-avm"
DEFAULT_REGISTERED_MODEL = "spec-nyc-avm"


@dataclass
class ArenaDecision:
    run_id: str
    model_version: str
    score: float
    gate_pass: bool
    weighted_segment_mdape_improvement: float
    overall_ppe10_lift: float
    max_major_segment_ppe10_drop: float
    min_major_segment_ppe10: float
    drift_alert_delta: int
    fairness_alert_delta: int
    tie_breaker_weighted_segment_ppe10: float
    tie_breaker_overall_mdape: float


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _json_load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Arena policy file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Policy file {path} is not JSON and pyyaml is unavailable. "
                "Install pyyaml or use JSON-formatted YAML."
            ) from exc
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Arena policy must load to dict. Got: {type(payload)}")
        return payload


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def proposal_is_expired(proposal: Dict, *, now: Optional[datetime] = None) -> bool:
    if "expires_at_utc" not in proposal:
        return False
    now_utc = now or _utcnow()
    return now_utc >= _parse_utc(str(proposal["expires_at_utc"]))


def _major_segment_names(metrics: Dict, *, min_n: int) -> List[str]:
    out = []
    per_segment = metrics.get("per_segment", {}) or {}
    for segment, seg_vals in per_segment.items():
        if not isinstance(seg_vals, dict):
            continue
        if int(seg_vals.get("n", 0)) >= int(min_n):
            out.append(str(segment))
    return sorted(out)


def _weighted_segment_metric(metrics: Dict, metric_key: str, *, min_n: int) -> float:
    per_segment = metrics.get("per_segment", {}) or {}
    total_weight = 0.0
    weighted = 0.0
    for _, seg_vals in per_segment.items():
        if not isinstance(seg_vals, dict):
            continue
        n = int(seg_vals.get("n", 0))
        if n < min_n:
            continue
        value = float(seg_vals.get(metric_key, 0.0))
        weighted += n * value
        total_weight += n
    if total_weight <= 0:
        return float(metrics.get("overall", {}).get(metric_key, 0.0))
    return weighted / total_weight


def _segment_ppe10_gap(metrics: Dict, *, min_n: int) -> float:
    values = []
    per_segment = metrics.get("per_segment", {}) or {}
    for _, seg_vals in per_segment.items():
        if not isinstance(seg_vals, dict):
            continue
        n = int(seg_vals.get("n", 0))
        if n < min_n:
            continue
        values.append(float(seg_vals.get("ppe10", 0.0)))
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _drift_alerts(metrics: Dict) -> int:
    meta = metrics.get("metadata", {}) or {}
    try:
        return int(meta.get("drift_alerts", 0))
    except Exception:
        return 0


def _fairness_alerts(metrics: Dict, *, min_n: int, max_gap: float) -> int:
    return int(_segment_ppe10_gap(metrics, min_n=min_n) > max_gap)


def _major_segment_drop(
    champion_metrics: Dict,
    challenger_metrics: Dict,
    *,
    min_n: int,
) -> float:
    champ = champion_metrics.get("per_segment", {}) or {}
    chal = challenger_metrics.get("per_segment", {}) or {}
    max_drop = 0.0
    for segment, champ_vals in champ.items():
        if not isinstance(champ_vals, dict):
            continue
        n = int(champ_vals.get("n", 0))
        if n < min_n:
            continue
        if segment not in chal or not isinstance(chal.get(segment), dict):
            continue
        champ_ppe10 = float(champ_vals.get("ppe10", 0.0))
        chal_ppe10 = float(chal[segment].get("ppe10", 0.0))
        max_drop = max(max_drop, champ_ppe10 - chal_ppe10)
    return max_drop


def _major_segment_floor(challenger_metrics: Dict, *, min_n: int) -> float:
    per_segment = challenger_metrics.get("per_segment", {}) or {}
    ppe10s = []
    for _, seg_vals in per_segment.items():
        if not isinstance(seg_vals, dict):
            continue
        if int(seg_vals.get("n", 0)) >= min_n:
            ppe10s.append(float(seg_vals.get("ppe10", 0.0)))
    if not ppe10s:
        return float(challenger_metrics.get("overall", {}).get("ppe10", 0.0))
    return min(ppe10s)


def evaluate_candidate(
    *,
    champion_metrics: Dict,
    challenger_metrics: Dict,
    challenger_run_id: str,
    challenger_model_version: str,
    policy: Dict,
) -> ArenaDecision:
    gates = policy.get("gates", {})
    scoring = policy.get("scoring", {})
    fairness = policy.get("fairness", {})
    min_n = int(policy.get("major_segment_min_n", 2000))

    mdape_improvement_threshold = float(gates.get("weighted_segment_mdape_improvement", 0.05))
    max_ppe10_drop_threshold = float(gates.get("max_major_segment_ppe10_drop", 0.02))
    major_segment_ppe10_floor = float(gates.get("major_segment_ppe10_floor", 0.24))
    max_segment_ppe10_gap = float(fairness.get("max_segment_ppe10_gap", 0.20))

    champ_w_mdape = _weighted_segment_metric(champion_metrics, "mdape", min_n=min_n)
    chal_w_mdape = _weighted_segment_metric(challenger_metrics, "mdape", min_n=min_n)
    weighted_segment_mdape_improvement = 0.0
    if champ_w_mdape > 0:
        weighted_segment_mdape_improvement = (champ_w_mdape - chal_w_mdape) / champ_w_mdape

    champ_overall_ppe10 = float(champion_metrics.get("overall", {}).get("ppe10", 0.0))
    chal_overall_ppe10 = float(challenger_metrics.get("overall", {}).get("ppe10", 0.0))
    overall_ppe10_lift = chal_overall_ppe10 - champ_overall_ppe10

    max_major_segment_ppe10_drop = _major_segment_drop(
        champion_metrics,
        challenger_metrics,
        min_n=min_n,
    )
    min_major_segment_ppe10 = _major_segment_floor(challenger_metrics, min_n=min_n)

    champ_drift_alerts = _drift_alerts(champion_metrics)
    chal_drift_alerts = _drift_alerts(challenger_metrics)
    drift_alert_delta = chal_drift_alerts - champ_drift_alerts

    champ_fair_alerts = _fairness_alerts(champion_metrics, min_n=min_n, max_gap=max_segment_ppe10_gap)
    chal_fair_alerts = _fairness_alerts(challenger_metrics, min_n=min_n, max_gap=max_segment_ppe10_gap)
    fairness_alert_delta = chal_fair_alerts - champ_fair_alerts

    gate_mdape = weighted_segment_mdape_improvement >= mdape_improvement_threshold
    gate_drop = max_major_segment_ppe10_drop <= max_ppe10_drop_threshold
    gate_floor = min_major_segment_ppe10 >= major_segment_ppe10_floor
    gate_drift = drift_alert_delta <= 0
    gate_fairness = fairness_alert_delta <= 0
    gate_pass = all([gate_mdape, gate_drop, gate_floor, gate_drift, gate_fairness])

    mdape_lift_norm = _clamp(
        weighted_segment_mdape_improvement / max(mdape_improvement_threshold, 1e-6),
        -1.0,
        1.5,
    )
    ppe10_lift_norm = _clamp(overall_ppe10_lift / 0.05, -1.0, 1.5)

    stability_bonus = 1.0
    if max_major_segment_ppe10_drop > 0:
        stability_bonus -= _clamp(max_major_segment_ppe10_drop / max(max_ppe10_drop_threshold * 2, 1e-6), 0.0, 0.6)
    if drift_alert_delta > 0:
        stability_bonus -= 0.2
    if fairness_alert_delta > 0:
        stability_bonus -= 0.2
    stability_bonus = _clamp(stability_bonus, 0.0, 1.0)

    score = (
        float(scoring.get("mdape_weight", 0.50)) * mdape_lift_norm
        + float(scoring.get("ppe10_weight", 0.30)) * ppe10_lift_norm
        + float(scoring.get("stability_weight", 0.20)) * stability_bonus
    )

    tie_breaker_weighted_segment_ppe10 = _weighted_segment_metric(challenger_metrics, "ppe10", min_n=min_n)
    tie_breaker_overall_mdape = float(challenger_metrics.get("overall", {}).get("mdape", 0.0))

    return ArenaDecision(
        run_id=challenger_run_id,
        model_version=challenger_model_version,
        score=float(score),
        gate_pass=gate_pass,
        weighted_segment_mdape_improvement=float(weighted_segment_mdape_improvement),
        overall_ppe10_lift=float(overall_ppe10_lift),
        max_major_segment_ppe10_drop=float(max_major_segment_ppe10_drop),
        min_major_segment_ppe10=float(min_major_segment_ppe10),
        drift_alert_delta=int(drift_alert_delta),
        fairness_alert_delta=int(fairness_alert_delta),
        tie_breaker_weighted_segment_ppe10=float(tie_breaker_weighted_segment_ppe10),
        tie_breaker_overall_mdape=float(tie_breaker_overall_mdape),
    )


def _sort_decisions(decisions: Iterable[ArenaDecision]) -> List[ArenaDecision]:
    return sorted(
        decisions,
        key=lambda d: (
            int(d.gate_pass),
            d.score,
            d.tie_breaker_weighted_segment_ppe10,
            -d.tie_breaker_overall_mdape,
        ),
        reverse=True,
    )


def _is_smoke_name(name: str) -> bool:
    lowered = (name or "").lower()
    return "smoke" in lowered or "dryrun" in lowered


def is_eligible_run(run_info: Dict) -> bool:
    status = str(run_info.get("status", "")).upper()
    if status != "FINISHED":
        return False
    run_name = str(run_info.get("run_name", ""))
    if _is_smoke_name(run_name):
        return False
    tags = run_info.get("tags", {}) or {}
    run_kind = str(tags.get("run_kind", "train"))
    if run_kind != "train":
        return False
    required_tags = ["hypothesis_id", "change_type", "change_summary", "owner", "feature_set_version", "dataset_version"]
    return all(bool(tags.get(tag)) for tag in required_tags)


def validate_change_note(note: Dict) -> Tuple[bool, List[str]]:
    required = [
        "problem_statement",
        "change_rationale",
        "change_details",
        "before_after_metrics",
        "risk_callouts",
        "rollback_pointer",
    ]
    missing = [field for field in required if not note.get(field)]
    return len(missing) == 0, missing


def _mlflow_client(tracking_uri: Optional[str] = None):
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    return mlflow, mlflow.tracking.MlflowClient()


def _find_metrics_artifact(client, run_id: str) -> Optional[str]:
    paths = []
    for item in client.list_artifacts(run_id, "metrics"):
        if item.path.endswith(".json"):
            paths.append(item.path)
    if not paths:
        return None
    # Prefer canonical metrics_v*.json over eval/smoke if present.
    preferred = [p for p in paths if "_eval" not in p and "smoke" not in p and "dryrun" not in p]
    return sorted(preferred or paths)[-1]


def _load_run_metrics(client, run_id: str) -> Dict:
    artifact_path = _find_metrics_artifact(client, run_id)
    if artifact_path is None:
        raise FileNotFoundError(f"No metrics JSON artifact found for run_id={run_id}")
    local_path = Path(client.download_artifacts(run_id, artifact_path))
    return _json_load(local_path)


def _get_alias_version(client, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None


def _proposal_path(proposal_id: str, arena_dir: Path) -> Path:
    return arena_dir / f"proposal_{proposal_id}.json"


def _latest_proposal(arena_dir: Path) -> Optional[Path]:
    proposals = sorted(arena_dir.glob("proposal_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return proposals[0] if proposals else None


def _append_model_change_log(entry: Dict, *, arena_dir: Path) -> Path:
    arena_dir.mkdir(parents=True, exist_ok=True)
    path = arena_dir / "model_change_log.md"
    if not path.exists():
        path.write_text(
            "\n".join(
                [
                    "# Model Change Log",
                    "",
                    "| Timestamp (UTC) | Proposal ID | Winner Run | Winner Version | Previous Champion | Decision | Summary |",
                    "|---|---|---|---|---|---|---|",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    line = (
        f"| {entry.get('timestamp_utc', '-')} | {entry.get('proposal_id', '-')} | {entry.get('winner_run_id', '-')} | "
        f"{entry.get('winner_model_version', '-')} | {entry.get('previous_champion_version', '-')} | "
        f"{entry.get('decision', '-')} | {str(entry.get('summary', '-')).replace('|', '/')} |"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return path


def _write_promotion_note(proposal: Dict, *, arena_dir: Path) -> Path:
    proposal_id = str(proposal.get("proposal_id", "unknown"))
    path = arena_dir / f"promotion_note_{proposal_id}.md"
    winner = proposal.get("winner", {}) or {}
    lines = [
        "# Promotion Note",
        "",
        f"- Proposal ID: `{proposal_id}`",
        f"- Decision: `{proposal.get('status', 'unknown')}`",
        f"- Approved By: `{proposal.get('approved_by', 'n/a')}`",
        f"- Previous Champion Version: `{proposal.get('previous_champion_version', 'n/a')}`",
        f"- Winner Run ID: `{winner.get('run_id', 'n/a')}`",
        f"- Winner Model Version: `{winner.get('model_version', 'n/a')}`",
        "",
        "## Before vs After (Selected Deltas)",
        f"- Weighted Segment MdAPE Improvement: `{winner.get('weighted_segment_mdape_improvement', 'n/a')}`",
        f"- Overall PPE10 Lift: `{winner.get('overall_ppe10_lift', 'n/a')}`",
        f"- Max Segment PPE10 Drop: `{winner.get('max_major_segment_ppe10_drop', 'n/a')}`",
        "",
        "## Rollback",
        f"- Rollback Pointer (previous champion version): `{proposal.get('previous_champion_version', 'n/a')}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_comparison_csv(decisions: List[ArenaDecision], *, arena_dir: Path) -> Path:
    arena_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = arena_dir / f"comparison_{timestamp}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "model_version",
                "gate_pass",
                "score",
                "weighted_segment_mdape_improvement",
                "overall_ppe10_lift",
                "max_major_segment_ppe10_drop",
                "min_major_segment_ppe10",
                "drift_alert_delta",
                "fairness_alert_delta",
                "tie_breaker_weighted_segment_ppe10",
                "tie_breaker_overall_mdape",
            ]
        )
        for d in decisions:
            writer.writerow(
                [
                    d.run_id,
                    d.model_version,
                    str(d.gate_pass).lower(),
                    f"{d.score:.6f}",
                    f"{d.weighted_segment_mdape_improvement:.6f}",
                    f"{d.overall_ppe10_lift:.6f}",
                    f"{d.max_major_segment_ppe10_drop:.6f}",
                    f"{d.min_major_segment_ppe10:.6f}",
                    d.drift_alert_delta,
                    d.fairness_alert_delta,
                    f"{d.tie_breaker_weighted_segment_ppe10:.6f}",
                    f"{d.tie_breaker_overall_mdape:.6f}",
                ]
            )
    return path


def _write_proposal_markdown(proposal: Dict, *, output_path: Path) -> None:
    winner = proposal.get("winner", {}) or {}
    lines = [
        "# Arena Promotion Proposal",
        "",
        f"- Proposal ID: `{proposal.get('proposal_id')}`",
        f"- Created (UTC): `{proposal.get('created_at_utc')}`",
        f"- Expires (UTC): `{proposal.get('expires_at_utc')}`",
        f"- Status: `{proposal.get('status')}`",
        f"- Promotion Mode: `{proposal.get('promotion_mode')}`",
        "",
        "## Champion",
        f"- Run ID: `{proposal.get('champion', {}).get('run_id', 'n/a')}`",
        f"- Model Version: `{proposal.get('champion', {}).get('model_version', 'n/a')}`",
        "",
        "## Winner Candidate",
        f"- Run ID: `{winner.get('run_id', 'n/a')}`",
        f"- Model Version: `{winner.get('model_version', 'n/a')}`",
        f"- Score: `{winner.get('score', 'n/a')}`",
        f"- Weighted Segment MdAPE Improvement: `{winner.get('weighted_segment_mdape_improvement', 'n/a')}`",
        f"- Overall PPE10 Lift: `{winner.get('overall_ppe10_lift', 'n/a')}`",
        "",
        f"## Comparison CSV",
        f"- `{proposal.get('comparison_csv', 'n/a')}`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _proposal_decision(
    *,
    policy: Dict,
    champion_metrics: Dict,
    champion_run_id: str,
    champion_model_version: str,
    candidate_payloads: List[Tuple[str, str, Dict]],
) -> Tuple[List[ArenaDecision], Optional[ArenaDecision]]:
    decisions: List[ArenaDecision] = []
    for run_id, model_version, metrics in candidate_payloads:
        decisions.append(
            evaluate_candidate(
                champion_metrics=champion_metrics,
                challenger_metrics=metrics,
                challenger_run_id=run_id,
                challenger_model_version=model_version,
                policy=policy,
            )
        )
    ranked = _sort_decisions(decisions)
    winner = ranked[0] if ranked and ranked[0].gate_pass else None
    return ranked, winner


def propose(
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: Optional[str] = None,
    selection_window: Optional[int] = None,
    arena_dir: Path = ARENA_DIR,
    set_aliases: bool = True,
) -> Dict:
    policy = load_policy(policy_path)
    model_name = registered_model_name or str(policy.get("registered_model_name", DEFAULT_REGISTERED_MODEL))
    candidate_limit = int(selection_window or policy.get("selection_window", 5))

    mlflow, client = _mlflow_client(tracking_uri=tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    champion_version = _get_alias_version(client, model_name, "champion")
    if champion_version is None:
        raise ValueError(
            f"No champion alias found for model '{model_name}'. "
            "Register a model and set alias 'champion' first."
        )

    champion_run_id = str(champion_version.run_id)
    champion_model_version = str(champion_version.version)
    champion_metrics = _load_run_metrics(client, champion_run_id)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max(25, candidate_limit * 10),
        order_by=["attributes.start_time DESC"],
    )

    candidate_rows: List[Tuple[str, str, Dict]] = []
    for _, row in runs.iterrows():
        run_id = str(row.get("run_id"))
        if run_id == champion_run_id:
            continue
        run_info = {
            "status": row.get("status"),
            "run_name": row.get("tags.mlflow.runName", ""),
            "tags": {
                "run_kind": row.get("tags.run_kind"),
                "hypothesis_id": row.get("tags.hypothesis_id"),
                "change_type": row.get("tags.change_type"),
                "change_summary": row.get("tags.change_summary"),
                "owner": row.get("tags.owner"),
                "feature_set_version": row.get("tags.feature_set_version"),
                "dataset_version": row.get("tags.dataset_version"),
            },
        }
        if not is_eligible_run(run_info):
            continue
        try:
            metrics = _load_run_metrics(client, run_id)
        except Exception:
            continue
        model_version = str(metrics.get("metadata", {}).get("model_version", "unknown"))
        candidate_rows.append((run_id, model_version, metrics))
        if len(candidate_rows) >= candidate_limit:
            break

    ranked, winner = _proposal_decision(
        policy=policy,
        champion_metrics=champion_metrics,
        champion_run_id=champion_run_id,
        champion_model_version=champion_model_version,
        candidate_payloads=candidate_rows,
    )

    comparison_csv = _write_comparison_csv(ranked, arena_dir=arena_dir)
    proposal_id = uuid.uuid4().hex[:12]
    created = _utcnow()
    expires = created + timedelta(hours=int(policy.get("promotion", {}).get("proposal_expiry_hours", 24)))
    status = "pending" if winner is not None else "no_winner"

    proposal = {
        "proposal_id": proposal_id,
        "created_at_utc": _to_iso(created),
        "expires_at_utc": _to_iso(expires),
        "status": status,
        "promotion_mode": str(policy.get("promotion", {}).get("mode", "semi_auto")),
        "policy_path": str(policy_path),
        "registered_model_name": model_name,
        "experiment_name": experiment_name,
        "champion": {
            "run_id": champion_run_id,
            "model_version": champion_model_version,
        },
        "comparison_csv": str(comparison_csv),
        "winner": None,
        "candidates_ranked": [
            {
                "run_id": d.run_id,
                "model_version": d.model_version,
                "score": d.score,
                "gate_pass": d.gate_pass,
                "weighted_segment_mdape_improvement": d.weighted_segment_mdape_improvement,
                "overall_ppe10_lift": d.overall_ppe10_lift,
                "max_major_segment_ppe10_drop": d.max_major_segment_ppe10_drop,
                "min_major_segment_ppe10": d.min_major_segment_ppe10,
                "drift_alert_delta": d.drift_alert_delta,
                "fairness_alert_delta": d.fairness_alert_delta,
            }
            for d in ranked
        ],
    }
    if winner is not None:
        proposal["winner"] = {
            "run_id": winner.run_id,
            "model_version": winner.model_version,
            "score": winner.score,
            "weighted_segment_mdape_improvement": winner.weighted_segment_mdape_improvement,
            "overall_ppe10_lift": winner.overall_ppe10_lift,
            "max_major_segment_ppe10_drop": winner.max_major_segment_ppe10_drop,
            "min_major_segment_ppe10": winner.min_major_segment_ppe10,
            "drift_alert_delta": winner.drift_alert_delta,
            "fairness_alert_delta": winner.fairness_alert_delta,
        }
        if set_aliases:
            client.set_registered_model_alias(model_name, "candidate", str(winner.model_version))
            client.set_registered_model_alias(model_name, "challenger", str(winner.model_version))

    arena_dir.mkdir(parents=True, exist_ok=True)
    json_path = _proposal_path(proposal_id, arena_dir)
    md_path = arena_dir / f"proposal_{proposal_id}.md"
    json_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    _write_proposal_markdown(proposal, output_path=md_path)
    proposal["proposal_json"] = str(json_path)
    proposal["proposal_md"] = str(md_path)
    return proposal


def approve(
    *,
    proposal_path: Optional[Path] = None,
    proposal_id: Optional[str] = None,
    arena_dir: Path = ARENA_DIR,
    tracking_uri: Optional[str] = None,
    approved_by: str = "manual_approver",
) -> Dict:
    if proposal_path is None:
        if proposal_id:
            proposal_path = _proposal_path(proposal_id, arena_dir)
        else:
            proposal_path = _latest_proposal(arena_dir)
    if proposal_path is None or not proposal_path.exists():
        raise FileNotFoundError("No proposal file found to approve.")

    proposal = _json_load(proposal_path)
    if proposal.get("status") != "pending":
        raise ValueError(f"Proposal status is '{proposal.get('status')}', expected 'pending'.")
    if proposal_is_expired(proposal):
        proposal["status"] = "expired"
        proposal_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
        raise ValueError("Proposal is expired and cannot be approved.")

    winner = proposal.get("winner") or {}
    winner_version = str(winner.get("model_version", ""))
    if not winner_version:
        raise ValueError("Proposal has no winner model_version.")

    model_name = str(proposal.get("registered_model_name", DEFAULT_REGISTERED_MODEL))
    _, client = _mlflow_client(tracking_uri=tracking_uri)
    previous_champion = _get_alias_version(client, model_name, "champion")
    previous_champion_version = str(previous_champion.version) if previous_champion is not None else None

    client.set_registered_model_alias(model_name, "champion", winner_version)
    proposal["status"] = "approved"
    proposal["approved_at_utc"] = _to_iso(_utcnow())
    proposal["approved_by"] = approved_by
    proposal["previous_champion_version"] = previous_champion_version
    proposal_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    note_path = _write_promotion_note(proposal, arena_dir=arena_dir)
    log_path = _append_model_change_log(
        {
            "timestamp_utc": proposal.get("approved_at_utc"),
            "proposal_id": proposal.get("proposal_id"),
            "winner_run_id": winner.get("run_id"),
            "winner_model_version": winner.get("model_version"),
            "previous_champion_version": previous_champion_version,
            "decision": "approved",
            "summary": f"Promoted winner with score={winner.get('score')}",
        },
        arena_dir=arena_dir,
    )
    return {
        "status": "approved",
        "proposal_path": str(proposal_path),
        "promotion_note": str(note_path),
        "model_change_log": str(log_path),
        "previous_champion_version": previous_champion_version,
        "new_champion_version": winner_version,
    }


def reject(
    *,
    reason: str,
    proposal_path: Optional[Path] = None,
    proposal_id: Optional[str] = None,
    arena_dir: Path = ARENA_DIR,
    rejected_by: str = "manual_reviewer",
) -> Dict:
    if proposal_path is None:
        if proposal_id:
            proposal_path = _proposal_path(proposal_id, arena_dir)
        else:
            proposal_path = _latest_proposal(arena_dir)
    if proposal_path is None or not proposal_path.exists():
        raise FileNotFoundError("No proposal file found to reject.")

    proposal = _json_load(proposal_path)
    if proposal.get("status") != "pending":
        raise ValueError(f"Proposal status is '{proposal.get('status')}', expected 'pending'.")

    proposal["status"] = "rejected"
    proposal["rejected_at_utc"] = _to_iso(_utcnow())
    proposal["rejected_by"] = rejected_by
    proposal["rejection_reason"] = reason
    proposal_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    log_path = _append_model_change_log(
        {
            "timestamp_utc": proposal.get("rejected_at_utc"),
            "proposal_id": proposal.get("proposal_id"),
            "winner_run_id": (proposal.get("winner") or {}).get("run_id"),
            "winner_model_version": (proposal.get("winner") or {}).get("model_version"),
            "previous_champion_version": proposal.get("champion", {}).get("model_version"),
            "decision": "rejected",
            "summary": reason,
        },
        arena_dir=arena_dir,
    )
    return {
        "status": "rejected",
        "proposal_path": str(proposal_path),
        "model_change_log": str(log_path),
    }


def status(
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
    tracking_uri: Optional[str] = None,
    arena_dir: Path = ARENA_DIR,
) -> Dict:
    policy = load_policy(policy_path)
    model_name = str(policy.get("registered_model_name", DEFAULT_REGISTERED_MODEL))

    _, client = _mlflow_client(tracking_uri=tracking_uri)
    aliases = {}
    for alias in ("champion", "challenger", "candidate"):
        version = _get_alias_version(client, model_name, alias)
        aliases[alias] = {
            "model_version": str(version.version) if version else None,
            "run_id": str(version.run_id) if version else None,
        }

    latest = _latest_proposal(arena_dir)
    latest_payload = _json_load(latest) if latest and latest.exists() else None
    return {
        "registered_model_name": model_name,
        "aliases": aliases,
        "latest_proposal_path": str(latest) if latest else None,
        "latest_proposal": latest_payload,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Champion/Challenger arena workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    propose_parser = sub.add_parser("propose", help="Evaluate challengers and write proposal.")
    propose_parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    propose_parser.add_argument("--tracking-uri", type=str, default=None)
    propose_parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    propose_parser.add_argument("--registered-model-name", type=str, default=None)
    propose_parser.add_argument("--selection-window", type=int, default=None)
    propose_parser.add_argument("--arena-dir", type=Path, default=ARENA_DIR)
    propose_parser.add_argument("--no-set-aliases", action="store_true")

    approve_parser = sub.add_parser("approve", help="Approve latest/picked proposal and promote champion.")
    approve_parser.add_argument("--proposal-id", type=str, default=None)
    approve_parser.add_argument("--proposal-path", type=Path, default=None)
    approve_parser.add_argument("--arena-dir", type=Path, default=ARENA_DIR)
    approve_parser.add_argument("--tracking-uri", type=str, default=None)
    approve_parser.add_argument("--approved-by", type=str, default="manual_approver")

    reject_parser = sub.add_parser("reject", help="Reject latest/picked proposal.")
    reject_parser.add_argument("--reason", type=str, required=True)
    reject_parser.add_argument("--proposal-id", type=str, default=None)
    reject_parser.add_argument("--proposal-path", type=Path, default=None)
    reject_parser.add_argument("--arena-dir", type=Path, default=ARENA_DIR)
    reject_parser.add_argument("--rejected-by", type=str, default="manual_reviewer")

    status_parser = sub.add_parser("status", help="Show current arena aliases and latest proposal.")
    status_parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    status_parser.add_argument("--tracking-uri", type=str, default=None)
    status_parser.add_argument("--arena-dir", type=Path, default=ARENA_DIR)

    args = parser.parse_args()

    if args.command == "propose":
        out = propose(
            policy_path=args.policy_path,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            registered_model_name=args.registered_model_name,
            selection_window=args.selection_window,
            arena_dir=args.arena_dir,
            set_aliases=not args.no_set_aliases,
        )
    elif args.command == "approve":
        out = approve(
            proposal_id=args.proposal_id,
            proposal_path=args.proposal_path,
            arena_dir=args.arena_dir,
            tracking_uri=args.tracking_uri,
            approved_by=args.approved_by,
        )
    elif args.command == "reject":
        out = reject(
            reason=args.reason,
            proposal_id=args.proposal_id,
            proposal_path=args.proposal_path,
            arena_dir=args.arena_dir,
            rejected_by=args.rejected_by,
        )
    else:
        out = status(
            policy_path=args.policy_path,
            tracking_uri=args.tracking_uri,
            arena_dir=args.arena_dir,
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
