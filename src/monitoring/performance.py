"""Performance monitor for latest model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.evaluate import evaluate_predictions


def find_latest_predictions() -> Optional[Path]:
    files = sorted(Path("reports/model").glob("evaluation_predictions_*.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def evaluate_performance(
    predictions_df: pd.DataFrame,
    *,
    warn_ppe10: float = 0.75,
    critical_ppe10: float = 0.65,
    warn_mdape: float = 0.08,
    critical_mdape: float = 0.12,
) -> Dict[str, object]:
    metrics = evaluate_predictions(predictions_df)
    overall = metrics["overall"]

    status = "ok"
    if overall["ppe10"] < critical_ppe10 or overall["mdape"] > critical_mdape:
        status = "alert"
    elif overall["ppe10"] < warn_ppe10 or overall["mdape"] > warn_mdape:
        status = "warn"

    return {
        "status": status,
        "thresholds": {
            "warn_ppe10": warn_ppe10,
            "critical_ppe10": critical_ppe10,
            "warn_mdape": warn_mdape,
            "critical_mdape": critical_mdape,
        },
        "metrics": metrics,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Monitor model performance against latest ground truth.")
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--warn-ppe10", type=float, default=0.75)
    parser.add_argument("--critical-ppe10", type=float, default=0.65)
    parser.add_argument("--warn-mdape", type=float, default=0.08)
    parser.add_argument("--critical-mdape", type=float, default=0.12)
    parser.add_argument("--output-json", type=Path, default=Path("reports/monitoring/performance_latest.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/monitoring/performance_latest.md"))
    args = parser.parse_args()

    pred_path = args.predictions_csv or find_latest_predictions()
    if pred_path is None or not pred_path.exists():
        raise FileNotFoundError("No predictions CSV found. Provide --predictions-csv.")

    df = pd.read_csv(pred_path)
    report = evaluate_performance(
        df,
        warn_ppe10=args.warn_ppe10,
        critical_ppe10=args.critical_ppe10,
        warn_mdape=args.warn_mdape,
        critical_mdape=args.critical_mdape,
    )
    report["predictions_csv"] = str(pred_path)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["metrics"]["overall"]
    args.output_md.write_text(
        "\n".join(
            [
                "# Performance Monitor",
                "",
                f"- Status: `{report['status']}`",
                f"- Predictions: `{pred_path}`",
                f"- PPE10: `{overall['ppe10']:.4f}`",
                f"- MdAPE: `{overall['mdape']:.4f}`",
                f"- R2: `{overall['r2']:.4f}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    _cli()

