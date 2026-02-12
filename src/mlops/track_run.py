"""MLflow tracking utility for S.P.E.C. NYC model runs."""

from __future__ import annotations

import argparse
from datetime import datetime
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


def _load_metrics_payload(metrics_json: Path) -> Dict:
    if not metrics_json.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {metrics_json}")
    return json.loads(metrics_json.read_text(encoding="utf-8"))


def _write_run_card(
    *,
    run_id: str,
    run_name: str,
    run_kind: str,
    metrics: Dict,
    metadata: Dict,
    hypothesis_id: str | None,
    change_type: str | None,
    change_summary: str | None,
    owner: str | None,
    feature_set_version: str | None,
    dataset_version: str | None,
    model_uri: str | None,
    registered_model_name: str | None,
    model_version: str | None,
    alias: str | None,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_card_{run_id}.md"
    overall = metrics.get("overall", {})
    segment_rows = []
    for segment, values in sorted((metrics.get("per_segment") or {}).items()):
        if not isinstance(values, dict):
            continue
        segment_rows.append(
            f"| {segment} | {int(values.get('n', 0))} | {float(values.get('ppe10', 0.0)):.4f} | "
            f"{float(values.get('mdape', 0.0)):.4f} | {float(values.get('r2', 0.0)):.4f} |"
        )
    if not segment_rows:
        segment_rows.append("| - | - | - | - | - |")

    lines = [
        "# Model Run Card",
        "",
        "## Identity",
        f"- Run ID: `{run_id}`",
        f"- Run Name: `{run_name}`",
        f"- Run Kind: `{run_kind}`",
        f"- Timestamp (UTC): `{datetime.utcnow().isoformat()}`",
        "",
        "## Change Narrative",
        f"- Hypothesis ID: `{hypothesis_id or 'n/a'}`",
        f"- Change Type: `{change_type or 'n/a'}`",
        f"- Change Summary: `{change_summary or 'n/a'}`",
        f"- Owner: `{owner or 'n/a'}`",
        f"- Feature Set Version: `{feature_set_version or 'n/a'}`",
        f"- Dataset Version: `{dataset_version or 'n/a'}`",
        "",
        "## Model and Registry",
        f"- Model URI: `{model_uri or 'n/a'}`",
        f"- Registered Model: `{registered_model_name or 'n/a'}`",
        f"- Model Version: `{model_version or 'n/a'}`",
        f"- Alias: `{alias or 'n/a'}`",
        "",
        "## Headline Metrics",
        f"- PPE10: `{float(overall.get('ppe10', 0.0)):.4f}`",
        f"- MdAPE: `{float(overall.get('mdape', 0.0)):.4f}`",
        f"- R2: `{float(overall.get('r2', 0.0)):.4f}`",
        "",
        "## Segment Metrics",
        "| Segment | n | PPE10 | MdAPE | R2 |",
        "|---|---:|---:|---:|---:|",
        *segment_rows,
        "",
        "## Risk Notes",
        (
            f"- Segment PPE10 variance flag: `{metadata.get('segment_variance_flag_v2', 'n/a')}` "
            f"(variance={metadata.get('segment_ppe10_variance', 'n/a')})"
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _ensure_registered_model(client, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)


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
    hypothesis_id: Optional[str] = None,
    change_type: Optional[str] = None,
    change_summary: Optional[str] = None,
    owner: Optional[str] = None,
    feature_set_version: Optional[str] = None,
    register_model: bool = False,
    registered_model_name: str = "spec-nyc-avm",
    alias: Optional[str] = None,
    run_kind: str = "train",
    arena_dir: Path = Path("reports/arena"),
) -> Dict[str, str]:
    """
    Log run metadata, metrics, params, and artifacts to MLflow.
    """
    try:
        import mlflow
    except Exception as exc:
        raise RuntimeError(f"mlflow is not installed/available: {exc}") from exc

    metrics = _load_metrics_payload(metrics_json)
    if not model_artifact.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_artifact}")
    metadata = metrics.get("metadata", {})
    overall = metrics.get("overall", {})

    run_kind = str(run_kind).strip().lower()
    if run_kind not in {"train", "backtest", "arena_eval"}:
        raise ValueError(f"Unsupported run_kind: {run_kind}")
    if change_type is not None:
        valid_change_types = {"feature", "objective", "data", "architecture", "tuning"}
        if change_type not in valid_change_types:
            raise ValueError(f"change_type must be one of {sorted(valid_change_types)}")
    if alias is not None and alias not in {"candidate", "challenger", "champion"}:
        raise ValueError("alias must be one of: candidate, challenger, champion")
    if register_model and run_kind == "train":
        required_meta = {
            "hypothesis_id": hypothesis_id,
            "change_type": change_type,
            "change_summary": change_summary,
            "feature_set_version": feature_set_version,
            "dataset_version": dataset_version,
            "owner": owner,
        }
        missing = [k for k, v in required_meta.items() if not v]
        if missing:
            raise ValueError(f"register_model train runs require metadata fields: {missing}")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.set_experiment(experiment_name)
    run_name = run_name or f"train-{metadata.get('model_version', 'v1')}"

    model_version: Optional[str] = None
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_version", str(metadata.get("model_version", "v1")))
        mlflow.set_tag("dataset_version", dataset_version or "unknown")
        mlflow.set_tag("git_sha", git_sha or get_git_sha())
        mlflow.set_tag("run_kind", run_kind)
        if hypothesis_id:
            mlflow.set_tag("hypothesis_id", hypothesis_id)
        if change_type:
            mlflow.set_tag("change_type", change_type)
        if change_summary:
            mlflow.set_tag("change_summary", change_summary)
        if owner:
            mlflow.set_tag("owner", owner)
        if feature_set_version:
            mlflow.set_tag("feature_set_version", feature_set_version)

        for key in [
            "train_rows",
            "test_rows",
            "optuna_trials",
            "segment_ppe10_variance",
            "segment_variance_flag_v2",
            "model_strategy",
            "router_mode",
            "segment_model_count",
            "min_segment_rows",
        ]:
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
        router_columns = metadata.get("router_columns", [])
        if isinstance(router_columns, list) and router_columns:
            mlflow.log_param("router_columns", ",".join(str(col) for col in router_columns))

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

        model_uri = f"runs:/{run.info.run_id}/model"
        if register_model:
            client = mlflow.tracking.MlflowClient()
            _ensure_registered_model(client, registered_model_name)
            created = client.create_model_version(
                name=registered_model_name,
                source=model_uri,
                run_id=run.info.run_id,
            )
            model_version = str(created.version)
            if alias:
                client.set_registered_model_alias(registered_model_name, alias, model_version)

        run_card = _write_run_card(
            run_id=run.info.run_id,
            run_name=run_name,
            run_kind=run_kind,
            metrics=metrics,
            metadata=metadata,
            hypothesis_id=hypothesis_id,
            change_type=change_type,
            change_summary=change_summary,
            owner=owner,
            feature_set_version=feature_set_version,
            dataset_version=dataset_version,
            model_uri=model_uri,
            registered_model_name=registered_model_name if register_model else None,
            model_version=model_version,
            alias=alias,
            output_dir=arena_dir,
        )
        mlflow.log_artifact(str(run_card), artifact_path="arena")

        result = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "run_card": str(run_card),
        }
        if register_model:
            result["registered_model_name"] = registered_model_name
            result["model_version"] = model_version or ""
            if alias:
                result["alias"] = alias
        return result


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
    parser.add_argument("--hypothesis-id", type=str, default=None)
    parser.add_argument("--change-type", type=str, default=None, choices=["feature", "objective", "data", "architecture", "tuning"])
    parser.add_argument("--change-summary", type=str, default=None)
    parser.add_argument("--owner", type=str, default=None)
    parser.add_argument("--feature-set-version", type=str, default=None)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--registered-model-name", type=str, default="spec-nyc-avm")
    parser.add_argument("--alias", type=str, default=None, choices=["candidate", "challenger", "champion"])
    parser.add_argument("--run-kind", type=str, default="train", choices=["train", "backtest", "arena_eval"])
    parser.add_argument("--arena-dir", type=Path, default=Path("reports/arena"))
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
        hypothesis_id=args.hypothesis_id,
        change_type=args.change_type,
        change_summary=args.change_summary,
        owner=args.owner,
        feature_set_version=args.feature_set_version,
        register_model=args.register_model,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
        run_kind=args.run_kind,
        arena_dir=args.arena_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
