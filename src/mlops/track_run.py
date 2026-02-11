"""MLflow tracking utility for S.P.E.C. NYC model runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional


def get_git_sha() -> str:
    """Return short git SHA or 'unknown'."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def log_run(
    *,
    metrics_json: Path,
    model_artifact: Path,
    scorecard_csv: Optional[Path] = None,
    predictions_csv: Optional[Path] = None,
    experiment_name: str = "spec-nyc-avm",
    run_name: Optional[str] = None,
    dataset_version: Optional[str] = None,
    git_sha: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Dict[str, str]:
    """
    Log run metadata, metrics, params, and artifacts to MLflow.
    """
    try:
        import mlflow
    except Exception as exc:
        raise RuntimeError(f"mlflow is not installed/available: {exc}") from exc

    if not metrics_json.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {metrics_json}")
    if not model_artifact.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_artifact}")

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    metadata = metrics.get("metadata", {})
    overall = metrics.get("overall", {})

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.set_experiment(experiment_name)
    run_name = run_name or f"train-{metadata.get('model_version', 'v1')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_version", str(metadata.get("model_version", "v1")))
        mlflow.set_tag("dataset_version", dataset_version or "unknown")
        mlflow.set_tag("git_sha", git_sha or get_git_sha())

        for key in ["train_rows", "test_rows", "optuna_trials", "segment_ppe10_variance", "segment_variance_flag_v2"]:
            if key in metadata and metadata[key] is not None:
                val = metadata[key]
                if isinstance(val, bool):
                    mlflow.log_param(key, str(val).lower())
                else:
                    mlflow.log_param(key, val)

        features = metadata.get("feature_columns", [])
        if isinstance(features, list):
            mlflow.log_param("feature_count", len(features))
            mlflow.log_param("feature_columns", ",".join(features))

        if isinstance(metadata.get("best_params"), dict):
            for k, v in metadata["best_params"].items():
                mlflow.log_param(f"best_{k}", v)

        for metric_key in ["ppe10", "mdape", "r2"]:
            if metric_key in overall:
                mlflow.log_metric(metric_key, float(overall[metric_key]))

        for segment, seg_metrics in metrics.get("per_segment", {}).items():
            if isinstance(seg_metrics, dict):
                if "ppe10" in seg_metrics:
                    mlflow.log_metric(f"segment_{segment}_ppe10", float(seg_metrics["ppe10"]))
                if "mdape" in seg_metrics:
                    mlflow.log_metric(f"segment_{segment}_mdape", float(seg_metrics["mdape"]))

        mlflow.log_artifact(str(metrics_json), artifact_path="metrics")
        mlflow.log_artifact(str(model_artifact), artifact_path="model")
        if scorecard_csv and scorecard_csv.exists():
            mlflow.log_artifact(str(scorecard_csv), artifact_path="reports")
        if predictions_csv and predictions_csv.exists():
            mlflow.log_artifact(str(predictions_csv), artifact_path="reports")

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": mlflow.get_tracking_uri(),
        }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Track a training/evaluation run to MLflow.")
    parser.add_argument("--metrics-json", type=Path, default=Path("models/metrics_v1.json"))
    parser.add_argument("--model-artifact", type=Path, default=Path("models/model_v1.joblib"))
    parser.add_argument("--scorecard-csv", type=Path, default=Path("reports/model/segment_scorecard_v1.csv"))
    parser.add_argument("--predictions-csv", type=Path, default=Path("reports/model/evaluation_predictions_v1.csv"))
    parser.add_argument("--experiment-name", type=str, default="spec-nyc-avm")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset-version", type=str, default=None)
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    result = log_run(
        metrics_json=args.metrics_json,
        model_artifact=args.model_artifact,
        scorecard_csv=args.scorecard_csv,
        predictions_csv=args.predictions_csv,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        dataset_version=args.dataset_version,
        git_sha=args.git_sha,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()

