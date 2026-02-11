"""Retrain decision policy for S.P.E.C. NYC."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_age_days(metrics_json: Dict) -> Optional[int]:
    trained_at = metrics_json.get("metadata", {}).get("trained_at_utc")
    if not trained_at:
        return None
    dt = pd.to_datetime(trained_at, errors="coerce")
    if pd.isna(dt):
        return None
    return int((datetime.utcnow() - dt.to_pydatetime()).days)


def evaluate_retrain_policy(
    *,
    metrics_json: Dict,
    performance_json: Dict,
    drift_df: pd.DataFrame,
    max_model_age_days: int = 90,
    min_ppe10: float = 0.75,
    max_mdape: float = 0.08,
    max_drift_alerts: int = 0,
) -> Dict[str, object]:
    reasons: List[str] = []

    overall = metrics_json.get("overall", {})
    perf_status = performance_json.get("status")
    if perf_status == "alert":
        reasons.append("performance monitor is alert")
    if "ppe10" in overall and float(overall["ppe10"]) < min_ppe10:
        reasons.append(f"overall PPE10 below threshold ({overall['ppe10']:.4f} < {min_ppe10:.4f})")
    if "mdape" in overall and float(overall["mdape"]) > max_mdape:
        reasons.append(f"overall MdAPE above threshold ({overall['mdape']:.4f} > {max_mdape:.4f})")

    drift_alerts = int((drift_df["status"] == "alert").sum()) if not drift_df.empty and "status" in drift_df.columns else 0
    if drift_alerts > max_drift_alerts:
        reasons.append(f"drift alerts exceed policy ({drift_alerts} > {max_drift_alerts})")

    age_days = _model_age_days(metrics_json)
    if age_days is not None and age_days > max_model_age_days:
        reasons.append(f"model age exceeds threshold ({age_days} > {max_model_age_days} days)")

    should_retrain = len(reasons) > 0
    return {
        "should_retrain": should_retrain,
        "decision": "retrain" if should_retrain else "hold",
        "reasons": reasons,
        "policy": {
            "max_model_age_days": max_model_age_days,
            "min_ppe10": min_ppe10,
            "max_mdape": max_mdape,
            "max_drift_alerts": max_drift_alerts,
        },
        "signals": {
            "drift_alerts": drift_alerts,
            "performance_status": perf_status,
            "model_age_days": age_days,
        },
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrain trigger policy.")
    parser.add_argument("--metrics-json", type=Path, default=Path("models/metrics_v1.json"))
    parser.add_argument("--performance-json", type=Path, default=Path("reports/monitoring/performance_latest.json"))
    parser.add_argument("--drift-csv", type=Path, default=Path("reports/monitoring/drift_latest.csv"))
    parser.add_argument("--max-model-age-days", type=int, default=90)
    parser.add_argument("--min-ppe10", type=float, default=0.75)
    parser.add_argument("--max-mdape", type=float, default=0.08)
    parser.add_argument("--max-drift-alerts", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=Path("reports/releases/retrain_decision_latest.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/releases/retrain_decision_latest.md"))
    args = parser.parse_args()

    metrics = _load_json(args.metrics_json)
    perf = _load_json(args.performance_json)
    drift_df = pd.read_csv(args.drift_csv) if args.drift_csv.exists() else pd.DataFrame()

    decision = evaluate_retrain_policy(
        metrics_json=metrics,
        performance_json=perf,
        drift_df=drift_df,
        max_model_age_days=args.max_model_age_days,
        min_ppe10=args.min_ppe10,
        max_mdape=args.max_mdape,
        max_drift_alerts=args.max_drift_alerts,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    args.output_md.write_text(
        "\n".join(
            [
                "# Retrain Decision",
                "",
                f"- Decision: `{decision['decision']}`",
                f"- Should Retrain: `{decision['should_retrain']}`",
                f"- Reasons: `{'; '.join(decision['reasons']) if decision['reasons'] else 'none'}`",
                f"- Signals: `{decision['signals']}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    _cli()

