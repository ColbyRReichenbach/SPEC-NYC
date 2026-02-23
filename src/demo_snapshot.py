from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _is_smoke_artifact(path: Path) -> bool:
    name = path.name.lower()
    return "smoke" in name or "dryrun" in name


def _latest_file(pattern: str, *, prefer_production: bool = True) -> Optional[Path]:
    files = list(Path(".").glob(pattern))
    if not files:
        return None
    if not prefer_production:
        return max(files, key=lambda p: p.stat().st_mtime)
    prod = [p for p in files if not _is_smoke_artifact(p)]
    if prod:
        return max(prod, key=lambda p: p.stat().st_mtime)
    return max(files, key=lambda p: p.stat().st_mtime)


def _read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_snapshot() -> Dict[str, Any]:
    metrics_path = _latest_file("models/metrics_*.json")
    model_path = _latest_file("models/model_*.joblib")
    scorecard_path = _latest_file("reports/model/segment_scorecard_*.csv")
    eval_path = _latest_file("reports/model/evaluation_predictions_*.csv")
    shap_summary_path = _latest_file("reports/model/shap_summary_*.png")
    shap_waterfall_path = _latest_file("reports/model/shap_waterfall_*.png")
    run_card_path = _latest_file("reports/arena/run_card_*.md")
    proposal_path = _latest_file("reports/arena/proposal_*.json")
    proposal_md_path = _latest_file("reports/arena/proposal_*.md")
    comparison_path = _latest_file("reports/arena/comparison_*.csv")
    drift_path = Path("reports/monitoring/drift_latest.json")
    performance_path = Path("reports/monitoring/performance_latest.json")
    retrain_path = Path("reports/releases/retrain_decision_latest.json")
    release_smoke_path = _latest_file("reports/validation/*smoke*.json", prefer_production=False)
    release_production_path = _latest_file("reports/validation/*production*.json", prefer_production=False)

    metrics = _read_json(metrics_path)
    proposal = _read_json(proposal_path)
    drift = _read_json(drift_path)
    performance = _read_json(performance_path)
    retrain = _read_json(retrain_path)
    smoke_release = _read_json(release_smoke_path)
    prod_release = _read_json(release_production_path)

    metadata = (metrics or {}).get("metadata", {})
    overall = (metrics or {}).get("overall", {})
    perf_overall = ((performance or {}).get("metrics", {}) or {}).get("overall", {})

    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "valuation_output_context": {
            "expected_output_fields": [
                "predicted_value",
                "low_value_bound",
                "high_value_bound",
                "confidence_score",
                "top_drivers",
            ],
            "latest_model_metrics": {
                "ppe10": overall.get("ppe10"),
                "mdape": overall.get("mdape"),
                "r2": overall.get("r2"),
                "n": overall.get("n"),
            },
        },
        "model_version_and_evidence": {
            "model_version": metadata.get("model_version"),
            "artifact_tag": metadata.get("artifact_tag"),
            "dataset_version": metadata.get("dataset_version"),
            "feature_set_version": metadata.get("feature_set_version"),
            "trained_at_utc": metadata.get("trained_at_utc"),
            "evidence_paths": {
                "metrics_json": str(metrics_path) if metrics_path else None,
                "model_joblib": str(model_path) if model_path else None,
                "segment_scorecard_csv": str(scorecard_path) if scorecard_path else None,
                "evaluation_predictions_csv": str(eval_path) if eval_path else None,
                "shap_summary_png": str(shap_summary_path) if shap_summary_path else None,
                "shap_waterfall_png": str(shap_waterfall_path) if shap_waterfall_path else None,
                "run_card_md": str(run_card_path) if run_card_path else None,
            },
        },
        "monitoring_snapshot": {
            "drift_status": (drift or {}).get("status"),
            "drift_alerts": (drift or {}).get("alerts"),
            "performance_status": (performance or {}).get("status"),
            "performance_overall": {
                "ppe10": perf_overall.get("ppe10"),
                "mdape": perf_overall.get("mdape"),
                "n": perf_overall.get("n"),
            },
            "retrain_decision": (retrain or {}).get("decision"),
            "retrain_reasons": (retrain or {}).get("reasons"),
            "evidence_paths": {
                "drift_json": str(drift_path) if drift_path.exists() else None,
                "performance_json": str(performance_path) if performance_path.exists() else None,
                "retrain_decision_json": str(retrain_path) if retrain_path.exists() else None,
            },
        },
        "promotion_decision_state": {
            "proposal_id": (proposal or {}).get("proposal_id"),
            "status": (proposal or {}).get("status"),
            "promotion_mode": (proposal or {}).get("promotion_mode"),
            "created_at_utc": (proposal or {}).get("created_at_utc"),
            "expires_at_utc": (proposal or {}).get("expires_at_utc"),
            "champion": (proposal or {}).get("champion"),
            "winner": (proposal or {}).get("winner"),
            "top_candidate": ((proposal or {}).get("candidates_ranked") or [None])[0],
            "evidence_paths": {
                "proposal_json": str(proposal_path) if proposal_path else None,
                "proposal_md": str(proposal_md_path) if proposal_md_path else None,
                "comparison_csv": str(comparison_path) if comparison_path else None,
            },
        },
        "release_validation_state": {
            "smoke_gate_e_green": (
                ((smoke_release or {}).get("gates", {}) or {}).get("Gate E (Release)", {})
            ).get("all_green"),
            "production_gate_e_green": (
                ((prod_release or {}).get("gates", {}) or {}).get("Gate E (Release)", {})
            ).get("all_green"),
            "evidence_paths": {
                "smoke_report_json": str(release_smoke_path) if release_smoke_path else None,
                "production_report_json": str(release_production_path) if release_production_path else None,
            },
        },
    }
    return snapshot


def write_markdown(snapshot: Dict[str, Any], output_md: Path) -> None:
    model = snapshot["model_version_and_evidence"]
    monitor = snapshot["monitoring_snapshot"]
    promo = snapshot["promotion_decision_state"]
    release = snapshot["release_validation_state"]
    valuation = snapshot["valuation_output_context"]

    lines = [
        "# Stakeholder Demo Snapshot",
        "",
        f"- Generated UTC: `{snapshot['generated_at_utc']}`",
        "",
        "## 1) Ingestion -> Valuation Output Context",
        "",
        f"- Output contract fields: `{', '.join(valuation['expected_output_fields'])}`",
        f"- Latest metrics: PPE10=`{valuation['latest_model_metrics'].get('ppe10')}`, MdAPE=`{valuation['latest_model_metrics'].get('mdape')}`, n=`{valuation['latest_model_metrics'].get('n')}`",
        "",
        "## 2) Model Version and Evidence",
        "",
        f"- model_version=`{model.get('model_version')}` | artifact_tag=`{model.get('artifact_tag')}`",
        f"- dataset_version=`{model.get('dataset_version')}` | feature_set_version=`{model.get('feature_set_version')}`",
        f"- trained_at_utc=`{model.get('trained_at_utc')}`",
    ]
    for key, value in model["evidence_paths"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## 3) Governance (Arena Promotion State)",
            "",
            f"- proposal_id=`{promo.get('proposal_id')}` | status=`{promo.get('status')}` | mode=`{promo.get('promotion_mode')}`",
            f"- champion=`{promo.get('champion')}`",
            f"- winner=`{promo.get('winner')}`",
            f"- top_candidate=`{promo.get('top_candidate')}`",
        ]
    )
    for key, value in promo["evidence_paths"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## 4) Monitoring Snapshot",
            "",
            f"- drift_status=`{monitor.get('drift_status')}` | drift_alerts=`{monitor.get('drift_alerts')}`",
            f"- performance_status=`{monitor.get('performance_status')}` | PPE10=`{monitor.get('performance_overall', {}).get('ppe10')}` | MdAPE=`{monitor.get('performance_overall', {}).get('mdape')}`",
            f"- retrain_decision=`{monitor.get('retrain_decision')}`",
            f"- retrain_reasons=`{monitor.get('retrain_reasons')}`",
        ]
    )
    for key, value in monitor["evidence_paths"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## 5) Release Gate State",
            "",
            f"- smoke_gate_e_green=`{release.get('smoke_gate_e_green')}`",
            f"- production_gate_e_green=`{release.get('production_gate_e_green')}`",
        ]
    )
    for key, value in release["evidence_paths"].items():
        lines.append(f"- {key}: `{value}`")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stakeholder demo snapshot from local artifacts.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/demo/stakeholder_demo_snapshot_latest.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/demo/stakeholder_demo_snapshot_latest.md"),
    )
    args = parser.parse_args()

    snapshot = build_snapshot()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    write_markdown(snapshot, args.output_md)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
