"""SHAP artifact generation for S.P.E.C. NYC models."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.inference import (
    build_route_keys,
    get_feature_columns,
    get_fallback_pipeline,
    get_model_strategy,
    get_router_columns,
    get_segment_pipelines,
)


logger = logging.getLogger(__name__)


def _normalize_artifact_tag(tag: str | None) -> str:
    raw = (tag or "").strip().lower()
    if not raw:
        return "prod"
    normalized = re.sub(r"[^a-z0-9_\-]", "_", raw).strip("_")
    return normalized or "prod"


def _artifact_stem(model_version: str, artifact_tag: str) -> str:
    normalized_tag = _normalize_artifact_tag(artifact_tag)
    return model_version if normalized_tag == "prod" else f"{model_version}_{normalized_tag}"


def _resolve_artifact_paths(
    *,
    model_version: str,
    artifact_tag: str,
    model_path: Path | None,
    evaluation_csv: Path | None,
    summary_plot_path: Path | None,
    waterfall_plot_path: Path | None,
) -> tuple[Path, Path, Path, Path]:
    stem = _artifact_stem(model_version, artifact_tag)
    resolved_model = model_path or Path(f"models/model_{stem}.joblib")
    resolved_eval = evaluation_csv or Path(f"reports/model/evaluation_predictions_{stem}.csv")
    resolved_summary = summary_plot_path or Path(f"reports/model/shap_summary_{stem}.png")
    resolved_waterfall = waterfall_plot_path or Path(f"reports/model/shap_waterfall_{stem}.png")
    return resolved_model, resolved_eval, resolved_summary, resolved_waterfall


def generate_shap_artifacts(
    *,
    model_path: Path,
    evaluation_csv: Path,
    summary_plot_path: Path,
    waterfall_plot_path: Path,
    sample_size: int = 500,
    random_state: int = 42,
) -> dict:
    """
    Generate SHAP summary and single-sample waterfall artifacts.

    Expects model artifact from `src.model` with keys:
    - pipeline
    - feature_columns
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not evaluation_csv.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {evaluation_csv}")

    artifact = joblib.load(model_path)
    feature_columns = get_feature_columns(artifact)
    if not feature_columns:
        raise ValueError("Model artifact missing feature_columns.")

    eval_df = pd.read_csv(evaluation_csv)
    missing_cols = [c for c in feature_columns if c not in eval_df.columns]
    if missing_cols:
        raise ValueError(f"Evaluation CSV missing required feature columns: {missing_cols}")

    strategy = get_model_strategy(artifact)
    segment_pipelines = get_segment_pipelines(artifact)
    router_columns = get_router_columns(artifact)
    explain_scope = "global"
    pipeline = get_fallback_pipeline(artifact)
    scope_df = eval_df

    has_router_col = any(col in eval_df.columns for col in router_columns)
    if strategy == "segmented_router" and has_router_col and segment_pipelines:
        route_values = build_route_keys(eval_df, router_columns)
        segment_counts = route_values.value_counts()
        chosen_segment = None
        for seg in segment_counts.index.tolist():
            if seg in segment_pipelines:
                chosen_segment = seg
                break
        if chosen_segment is not None:
            pipeline = segment_pipelines[chosen_segment]
            scope_df = eval_df.loc[route_values[route_values == chosen_segment].index].copy()
            explain_scope = f"route:{chosen_segment}"

    if pipeline is None:
        raise ValueError("Model artifact has no pipeline available for SHAP.")

    feature_frame = scope_df[feature_columns].copy()
    sampled_frame = feature_frame.sample(
        n=min(sample_size, len(feature_frame)),
        random_state=random_state,
    )

    transformed = pipeline.named_steps["preprocessor"].transform(sampled_frame)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    model = pipeline.named_steps["model"]

    explainer_type = "tree"
    base_values = None
    shap_exp = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_dense)
    except Exception as exc:
        logger.warning("TreeExplainer failed (%s). Trying xgboost pred_contribs fallback.", exc)
        try:
            import xgboost as xgb

            explainer_type = "xgboost_pred_contribs"
            dmatrix = xgb.DMatrix(transformed_dense, feature_names=list(pipeline.named_steps["preprocessor"].get_feature_names_out()))
            contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
            shap_values = contribs[:, :-1]
            base_values = contribs[:, -1]
        except Exception as xgb_exc:
            logger.warning("xgboost pred_contribs failed (%s). Falling back to model-agnostic SHAP Explainer.", xgb_exc)
            explainer_type = "agnostic"
            background = shap.sample(transformed_dense, min(100, len(transformed_dense)), random_state=random_state)
            explainer = shap.Explainer(
                model.predict,
                background,
                feature_names=pipeline.named_steps["preprocessor"].get_feature_names_out(),
            )
            shap_exp = explainer(transformed_dense)
            shap_values = shap_exp.values
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    summary_plot_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values,
        transformed_dense,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    row_idx = 0
    if explainer_type == "tree":
        expected_value = explainer.expected_value
        base_value = float(expected_value[0]) if isinstance(expected_value, (list, np.ndarray)) else float(expected_value)
        exp = shap.Explanation(
            values=shap_values[row_idx],
            base_values=base_value,
            data=transformed_dense[row_idx],
            feature_names=feature_names,
        )
    elif explainer_type == "xgboost_pred_contribs":
        exp = shap.Explanation(
            values=shap_values[row_idx],
            base_values=float(base_values[row_idx]),
            data=transformed_dense[row_idx],
            feature_names=feature_names,
        )
    else:
        exp = shap.Explanation(
            values=shap_values[row_idx],
            base_values=float(np.asarray(shap_exp.base_values[row_idx]).reshape(-1)[0]),
            data=transformed_dense[row_idx],
            feature_names=feature_names,
        )

    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(waterfall_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP summary plot: %s", summary_plot_path)
    logger.info("Saved SHAP waterfall plot: %s", waterfall_plot_path)
    return {
        "summary_plot_path": str(summary_plot_path),
        "waterfall_plot_path": str(waterfall_plot_path),
        "sample_size": int(len(sampled_frame)),
        "explainer_type": explainer_type,
        "explain_scope": explain_scope,
        "model_strategy": strategy,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts for trained model.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--evaluation-csv", type=Path, default=None)
    parser.add_argument("--summary-plot-path", type=Path, default=None)
    parser.add_argument("--waterfall-plot-path", type=Path, default=None)
    parser.add_argument("--model-version", type=str, default="v1")
    parser.add_argument(
        "--artifact-tag",
        type=str,
        default="prod",
        help="Artifact suffix tag. Use 'prod' for canonical paths; non-prod tags append _<tag>.",
    )
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    model_path, evaluation_csv, summary_plot_path, waterfall_plot_path = _resolve_artifact_paths(
        model_version=args.model_version,
        artifact_tag=args.artifact_tag,
        model_path=args.model_path,
        evaluation_csv=args.evaluation_csv,
        summary_plot_path=args.summary_plot_path,
        waterfall_plot_path=args.waterfall_plot_path,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    result = generate_shap_artifacts(
        model_path=model_path,
        evaluation_csv=evaluation_csv,
        summary_plot_path=summary_plot_path,
        waterfall_plot_path=waterfall_plot_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print(f"Generated SHAP artifacts: {result}")


if __name__ == "__main__":
    _cli()
