"""Inference helpers for global and segmented-router model artifacts."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

ROUTE_DELIMITER = "||"
MISSING_ROUTE_TOKEN = "__missing__"


def get_model_strategy(artifact: Dict[str, Any]) -> str:
    strategy = str(artifact.get("model_strategy", "global")).strip().lower()
    return strategy or "global"


def get_feature_columns(artifact: Dict[str, Any]) -> list[str]:
    raw = artifact.get("feature_columns", [])
    if not isinstance(raw, list):
        return []
    return [str(col) for col in raw]


def get_router_column(artifact: Dict[str, Any]) -> str:
    return str(artifact.get("router_column", "property_segment"))


def get_router_columns(artifact: Dict[str, Any]) -> list[str]:
    raw = artifact.get("router_columns", [])
    if isinstance(raw, list) and raw:
        return [str(col) for col in raw]
    return [get_router_column(artifact)]


def get_segment_pipelines(artifact: Dict[str, Any]) -> Dict[str, Any]:
    raw = artifact.get("segment_pipelines", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        out[str(key)] = value
    return out


def get_fallback_pipeline(artifact: Dict[str, Any]) -> Any:
    return artifact.get("fallback_pipeline") or artifact.get("pipeline")


def build_route_key(parts: list[Any]) -> str:
    normalized = []
    for value in parts:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            normalized.append(MISSING_ROUTE_TOKEN)
        else:
            text = str(value).strip()
            normalized.append(text or MISSING_ROUTE_TOKEN)
    return ROUTE_DELIMITER.join(normalized)


def build_route_key_from_row(row: pd.Series, router_columns: list[str]) -> str:
    values: list[Any] = []
    for col in router_columns:
        if col not in row.index:
            values.append(MISSING_ROUTE_TOKEN)
        else:
            values.append(row.get(col))
    return build_route_key(values)


def build_route_keys(frame: pd.DataFrame, router_columns: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series([], index=frame.index, dtype=str)
    parts = []
    for col in router_columns:
        if col in frame.columns:
            part = frame[col]
        else:
            part = pd.Series([MISSING_ROUTE_TOKEN] * len(frame), index=frame.index)
        parts.append(
            part.apply(
                lambda value: (
                    MISSING_ROUTE_TOKEN
                    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""
                    else str(value).strip()
                )
            )
        )
    if len(parts) == 1:
        return parts[0].astype(str)
    route_series = parts[0].astype(str)
    for part in parts[1:]:
        route_series = route_series + ROUTE_DELIMITER + part.astype(str)
    return route_series.astype(str)


def select_pipeline_for_route(
    artifact: Dict[str, Any],
    route_key: str,
) -> Tuple[Any, str]:
    strategy = get_model_strategy(artifact)
    segment_pipelines = get_segment_pipelines(artifact)
    fallback = get_fallback_pipeline(artifact)

    key = str(route_key or "")
    if strategy == "segmented_router" and key in segment_pipelines:
        return segment_pipelines[key], f"route:{key}"

    if fallback is None:
        raise ValueError("Model artifact has no fallback/global pipeline.")
    if strategy == "segmented_router":
        return fallback, "fallback_global"
    return fallback, "global"


def select_pipeline_for_row(artifact: Dict[str, Any], row: pd.Series) -> Tuple[Any, str]:
    router_columns = get_router_columns(artifact)
    route_key = build_route_key_from_row(row, router_columns)
    return select_pipeline_for_route(artifact, route_key)


def select_pipeline_for_segment(
    artifact: Dict[str, Any],
    segment_value: Any,
) -> Tuple[Any, str]:
    """Backward-compatible single-segment selector."""
    return select_pipeline_for_route(artifact, build_route_key([segment_value]))


def predict_dataframe(
    artifact: Dict[str, Any],
    frame: pd.DataFrame,
) -> Tuple[np.ndarray, pd.Series]:
    if frame.empty:
        empty_idx = frame.index
        return np.asarray([], dtype=float), pd.Series([], index=empty_idx, dtype=str)

    feature_columns = get_feature_columns(artifact)
    if not feature_columns:
        raise ValueError("Model artifact missing feature_columns.")
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Inference frame missing required columns: {missing}")

    strategy = get_model_strategy(artifact)
    router_columns = get_router_columns(artifact)
    segment_pipelines = get_segment_pipelines(artifact)
    fallback = get_fallback_pipeline(artifact)
    if fallback is None:
        raise ValueError("Model artifact has no fallback/global pipeline.")

    x = frame[feature_columns]
    has_any_router_col = any(col in frame.columns for col in router_columns)
    if strategy != "segmented_router" or not segment_pipelines or not has_any_router_col:
        preds = np.asarray(fallback.predict(x), dtype=float)
        label = "global" if strategy == "global" else "fallback_global"
        routes = pd.Series(label, index=frame.index, dtype=str)
        return preds, routes

    preds = pd.Series(np.nan, index=frame.index, dtype=float)
    routes = pd.Series("", index=frame.index, dtype=str)
    route_values = build_route_keys(frame, router_columns)

    for route_value in sorted(route_values.unique().tolist()):
        idx = route_values[route_values == route_value].index
        pipeline, route_label = select_pipeline_for_route(artifact, route_value)
        preds.loc[idx] = np.asarray(pipeline.predict(x.loc[idx]), dtype=float)
        routes.loc[idx] = route_label

    return preds.to_numpy(dtype=float), routes


def predict_single_row(artifact: Dict[str, Any], row: pd.Series) -> Tuple[float, str]:
    one = pd.DataFrame([row.to_dict()])
    preds, routes = predict_dataframe(artifact, one)
    return float(preds[0]), str(routes.iloc[0])
