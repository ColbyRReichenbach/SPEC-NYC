"""Inference helpers for global and segmented-router model artifacts."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.price_tier_proxy import assign_price_tier_proxy

ROUTE_DELIMITER = "||"
MISSING_ROUTE_TOKEN = "__missing__"
TREND_BASE_DATE = pd.Timestamp("2019-01-01")

# Features/keys must be inference-available and non-target-derived.
FORBIDDEN_TARGET_DERIVED_COLUMNS = {
    "sale_price",
    "price_tier",
    "predicted_price",
    "prediction_error",
    "abs_pct_error",
    "sale_price_true",
    "sale_price_pred",
    "price_per_sqft",
    "price_change_pct",
    "previous_sale_price",
    "previous_sale_date",
    "days_since_last_sale",
    "sale_sequence",
    "is_latest_sale",
    "target",
    "y",
}
INFERENCE_AVAILABLE_COLUMNS = {
    "gross_square_feet",
    "year_built",
    "building_age",
    "residential_units",
    "total_units",
    "distance_to_center_km",
    "h3_prior_sale_count",
    "h3_prior_median_price",
    "h3_prior_median_ppsf",
    "comp_count",
    "comp_median_price",
    "comp_median_ppsf",
    "comp_weighted_estimate",
    "comp_price_dispersion",
    "comp_nearest_distance_km",
    "comp_median_recency_days",
    "comp_local_momentum",
    "days_since_2019_start",
    "month_sin",
    "month_cos",
    "borough",
    "building_class",
    "property_segment",
    "neighborhood",
    "rate_regime_bucket",
    "price_tier_proxy",
}


def get_model_strategy(artifact: Dict[str, Any]) -> str:
    strategy = str(artifact.get("model_strategy", "global")).strip().lower()
    return strategy or "global"


def validate_feature_columns_for_inference(
    columns: list[str],
    *,
    context: str = "feature_columns",
) -> list[str]:
    normalized = [str(col).strip() for col in columns]
    lowered = [col.lower() for col in normalized]
    forbidden = sorted({col for col in lowered if col in FORBIDDEN_TARGET_DERIVED_COLUMNS})
    if forbidden:
        raise ValueError(f"{context} contains target-derived columns that are disallowed: {forbidden}")

    undocumented = sorted({col for col in normalized if col not in INFERENCE_AVAILABLE_COLUMNS})
    if undocumented:
        raise ValueError(
            f"{context} contains columns without inference-availability contract: {undocumented}. "
            "Add explicit inference-safe handling before training."
        )
    return normalized


def validate_router_columns_for_inference(
    columns: list[str],
    *,
    context: str = "router_columns",
) -> list[str]:
    out = validate_feature_columns_for_inference(columns, context=context)
    lowered = [c.lower() for c in out]
    if "price_tier" in lowered:
        raise ValueError(
            "Routing on target-derived 'price_tier' is disallowed. Use non-leaky 'price_tier_proxy' instead."
        )
    return out


def get_feature_columns(artifact: Dict[str, Any]) -> list[str]:
    raw = artifact.get("feature_columns", [])
    if not isinstance(raw, list):
        return []
    return validate_feature_columns_for_inference([str(col) for col in raw], context="feature_columns")


def get_router_column(artifact: Dict[str, Any]) -> str:
    return str(artifact.get("router_column", "property_segment"))


def get_router_columns(artifact: Dict[str, Any]) -> list[str]:
    raw = artifact.get("router_columns", [])
    if isinstance(raw, list) and raw:
        out = [str(col) for col in raw]
    else:
        out = [get_router_column(artifact)]
    return validate_router_columns_for_inference(out, context="router_columns")


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


def _with_router_columns(artifact: Dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure required routing columns are present using non-leaky fallback logic."""
    router_columns = get_router_columns(artifact)
    if "price_tier_proxy" not in router_columns or "price_tier_proxy" in frame.columns:
        return frame

    bins = artifact.get("price_tier_proxy_bins")
    if isinstance(bins, dict):
        out, _ = assign_price_tier_proxy(frame, bins=bins, segment_col="property_segment")
        return out

    # Last-resort fallback for legacy artifacts without bins.
    out = frame.copy()
    out["price_tier_proxy"] = "core"
    return out


def _with_temporal_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal features when artifacts require them."""
    out = frame.copy()
    required = {"days_since_2019_start", "month_sin", "month_cos", "rate_regime_bucket"}
    if required.issubset(set(out.columns)):
        return out

    sale_date = pd.to_datetime(out.get("sale_date"), errors="coerce") if "sale_date" in out.columns else pd.Series(pd.NaT, index=out.index)
    month = sale_date.dt.month.fillna(1).astype(float)

    if "days_since_2019_start" not in out.columns:
        out["days_since_2019_start"] = (sale_date - TREND_BASE_DATE).dt.days.astype("float64")
    if "month_sin" not in out.columns:
        out["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0).astype("float64")
    if "month_cos" not in out.columns:
        out["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0).astype("float64")
    if "rate_regime_bucket" not in out.columns:
        out["rate_regime_bucket"] = np.select(
            [
                sale_date < pd.Timestamp("2020-03-01"),
                sale_date < pd.Timestamp("2022-01-01"),
                sale_date < pd.Timestamp("2024-01-01"),
            ],
            [
                "pre_2020",
                "pandemic_low_rate",
                "post_hike_transition",
            ],
            default="high_rate_2024_plus",
        )
    return out


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
    row_df = pd.DataFrame([row.to_dict()])
    row_df = _with_router_columns(artifact, row_df)
    route_key = build_route_key_from_row(row_df.iloc[0], router_columns)
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

    frame = _with_router_columns(artifact, frame)
    frame = _with_temporal_regime_features(frame)

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
