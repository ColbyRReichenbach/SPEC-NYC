"""Deterministic as-of comparable-sales features for the NYC AVM.

The comps engine is intentionally rule-based and auditable. It selects only
historical valid sales before the valuation row date and emits both aggregate
model features and row-level comp evidence for review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.pre_comps import DATE_COL, TARGET_COL, stable_row_ids


COMP_FEATURES = [
    "comp_count",
    "comp_median_price",
    "comp_median_ppsf",
    "comp_weighted_estimate",
    "comp_price_dispersion",
    "comp_nearest_distance_km",
    "comp_median_recency_days",
    "comp_local_momentum",
]

SELECTED_COMP_COLUMNS = [
    "valuation_row_id",
    "valuation_split",
    "valuation_sale_date",
    "valuation_property_id",
    "comp_rank",
    "comp_row_id",
    "comp_property_id",
    "comp_sale_date",
    "comp_sale_price",
    "comp_ppsf",
    "comp_borough",
    "comp_property_segment",
    "comp_building_class",
    "comp_h3_index",
    "comp_distance_km",
    "comp_recency_days",
    "comp_score",
    "comp_weight",
    "eligibility_scope",
]


@dataclass(frozen=True)
class CompEngineConfig:
    """Comparable-sales rule configuration."""

    top_k: int = 8
    min_comps: int = 3
    primary_max_age_days: int = 730
    fallback_max_age_days: int = 1825
    max_distance_km: float = 3.0
    max_sqft_ratio_diff: float = 0.75
    max_age_diff_years: float = 50.0
    max_unit_diff: float = 10.0
    max_candidate_pool_per_group: int = 300
    max_candidate_pool_fallback: int = 600


@dataclass(frozen=True)
class CompsFeatureResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    selected_comps: pd.DataFrame
    manifest: dict[str, Any]


def add_asof_comps_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    config: CompEngineConfig | None = None,
    persist_selected_split: str = "test",
) -> CompsFeatureResult:
    """Add leakage-safe comparable-sales features to train and holdout frames."""
    cfg = config or CompEngineConfig()
    train = train_df.copy()
    test = test_df.copy()

    train_features, train_selected, train_manifest = compute_asof_comps_for_frame(
        train,
        train,
        split_name="train",
        config=cfg,
        include_selected=persist_selected_split in {"train", "both"},
    )
    test_features, test_selected, test_manifest = compute_asof_comps_for_frame(
        test,
        train,
        split_name="test",
        config=cfg,
        include_selected=persist_selected_split in {"test", "both"},
    )

    for feature in COMP_FEATURES:
        train[feature] = train_features[feature].reindex(train.index)
        test[feature] = test_features[feature].reindex(test.index)

    selected_parts = [frame for frame in (train_selected, test_selected) if not frame.empty]
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else empty_selected_comps_frame()
    if selected.empty:
        selected = empty_selected_comps_frame()

    manifest = {
        "feature_names": COMP_FEATURES,
        "config": asdict(cfg),
        "eligibility_policy": {
            "as_of_rule": "comp.sale_date < valuation.sale_date",
            "reference_policy": {
                "train_rows": "historical rows from the training split only",
                "test_rows": "training split rows only; holdout targets are never eligible comps",
            },
            "required_sale_validity_status": "valid_training_sale",
            "primary_filters": [
                "same borough",
                "same or compatible property segment",
                "within primary_max_age_days",
                "within max_distance_km when coordinates are available",
                "within max_sqft_ratio_diff when square footage is available",
                "within max_age_diff_years when building age is available",
                "within max_unit_diff when unit counts are available",
            ],
            "fallback_policy": (
                "If primary filters produce fewer than min_comps, keep same-borough compatible-segment "
                "historical candidates up to fallback_max_age_days and rank with explicit distance/feature penalties."
            ),
        },
        "train": train_manifest,
        "test": test_manifest,
        "selected_comps_rows": int(len(selected)),
        "selected_comps_schema_version": 1,
    }
    return CompsFeatureResult(train_df=train, test_df=test, selected_comps=selected, manifest=manifest)


def compute_asof_comps_for_frame(
    target_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    split_name: str,
    config: CompEngineConfig | None = None,
    include_selected: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute as-of comp features for one target frame from one reference frame."""
    cfg = config or CompEngineConfig()
    features = pd.DataFrame(index=target_df.index, columns=COMP_FEATURES, dtype="float64")
    features["comp_count"] = 0.0
    selected_rows: list[dict[str, Any]] = []

    targets = _prepare_targets(target_df)
    reference = _prepare_reference(reference_df)
    if targets.empty or reference.empty:
        manifest = _frame_manifest(features, selected_rows, cfg)
        return features, empty_selected_comps_frame(), manifest

    indexes = _build_reference_indexes(reference)
    for target_idx, target in targets.iterrows():
        selected, scope = _select_comps(target, indexes, cfg)
        row_features = _aggregate_features(target, selected)
        for feature, value in row_features.items():
            features.loc[target_idx, feature] = value
        if include_selected and not selected.empty:
            selected_rows.extend(_selected_comp_records(target, selected, split_name, scope))

    selected_frame = pd.DataFrame(selected_rows, columns=SELECTED_COMP_COLUMNS)
    if selected_frame.empty:
        selected_frame = empty_selected_comps_frame()
    manifest = _frame_manifest(features, selected_rows, cfg)
    return features, selected_frame, manifest


def empty_selected_comps_frame() -> pd.DataFrame:
    """Return an empty selected-comps evidence frame with a stable schema."""
    return pd.DataFrame(columns=SELECTED_COMP_COLUMNS)


def selected_comps_for_row(selected_comps: pd.DataFrame, row_id: str) -> pd.DataFrame:
    """Fetch persisted comps for a valuation row ID."""
    if selected_comps.empty or "valuation_row_id" not in selected_comps.columns:
        return empty_selected_comps_frame()
    out = selected_comps[selected_comps["valuation_row_id"].astype(str) == str(row_id)].copy()
    return out.sort_values("comp_rank").reset_index(drop=True)


def _prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_row_id"] = stable_row_ids(out)
    out[DATE_COL] = pd.to_datetime(out.get(DATE_COL), errors="coerce")
    out["_borough_key"] = _string_key(out.get("borough"))
    out["_segment_key"] = _string_key(out.get("property_segment"))
    out["_building_class_key"] = _string_key(out.get("building_class"))
    out["_h3_key"] = _string_key(out.get("h3_index"))
    for col in ["gross_square_feet", "building_age", "total_units", "residential_units", "latitude", "longitude"]:
        out[f"_{col}"] = _numeric_column(out, col)
    return out.dropna(subset=[DATE_COL])


def _prepare_reference(df: pd.DataFrame) -> pd.DataFrame:
    out = _prepare_targets(df)
    if "sale_validity_status" in out.columns:
        out = out[out["sale_validity_status"].astype("string") == "valid_training_sale"].copy()
    out[TARGET_COL] = pd.to_numeric(out.get(TARGET_COL), errors="coerce")
    out = out.dropna(subset=[DATE_COL, TARGET_COL])
    out = out[out[TARGET_COL] >= 10_000].copy()
    sqft = _numeric_column(out, "gross_square_feet")
    out["_comp_ppsf"] = out[TARGET_COL] / sqft.where(sqft > 0)
    out.loc[~np.isfinite(out["_comp_ppsf"]) | (out["_comp_ppsf"] <= 0), "_comp_ppsf"] = np.nan
    out = out.sort_values([DATE_COL, "_row_id"]).reset_index(drop=True)
    return out


def _build_reference_indexes(reference: pd.DataFrame) -> dict[str, dict[Any, pd.DataFrame]]:
    by_borough_segment: dict[Any, pd.DataFrame] = {}
    by_borough: dict[Any, pd.DataFrame] = {}
    for key, gdf in reference.groupby(["_borough_key", "_segment_key"], dropna=False):
        by_borough_segment[key] = gdf.sort_values([DATE_COL, "_row_id"]).reset_index(drop=True)
    for key, gdf in reference.groupby("_borough_key", dropna=False):
        by_borough[key] = gdf.sort_values([DATE_COL, "_row_id"]).reset_index(drop=True)
    return {"by_borough_segment": by_borough_segment, "by_borough": by_borough}


def _select_comps(
    target: pd.Series,
    indexes: dict[str, dict[Any, pd.DataFrame]],
    cfg: CompEngineConfig,
) -> tuple[pd.DataFrame, str]:
    borough = target.get("_borough_key")
    segment = target.get("_segment_key")
    valuation_date = target.get(DATE_COL)
    if pd.isna(valuation_date) or not borough:
        return pd.DataFrame(), "none"

    candidate_frames = []
    for candidate_segment in _compatible_segments(segment):
        group = indexes["by_borough_segment"].get((borough, candidate_segment))
        if group is None:
            continue
        candidate_frames.append(
            _prior_window(
                group,
                valuation_date,
                max_age_days=cfg.fallback_max_age_days,
                max_pool=cfg.max_candidate_pool_per_group,
            )
        )

    candidates = _concat_candidates(candidate_frames)
    if len(candidates) < cfg.min_comps:
        borough_group = indexes["by_borough"].get(borough)
        if borough_group is not None:
            candidates = _concat_candidates(
                [
                    candidates,
                    _prior_window(
                        borough_group,
                        valuation_date,
                        max_age_days=cfg.fallback_max_age_days,
                        max_pool=cfg.max_candidate_pool_fallback,
                    ),
                ]
            )

    if candidates.empty:
        return candidates, "none"

    scored = _score_candidates(target, candidates, cfg)
    if scored.empty:
        return scored, "none"

    primary = scored[scored["_primary_eligible"]].copy()
    if len(primary) >= cfg.min_comps:
        selected = primary.sort_values(["_comp_score", DATE_COL, "_row_id"]).head(cfg.top_k).copy()
        return _with_normalized_weights(selected), "primary"

    selected = scored.sort_values(["_comp_score", DATE_COL, "_row_id"]).head(cfg.top_k).copy()
    return _with_normalized_weights(selected), "fallback"


def _prior_window(group: pd.DataFrame, valuation_date: pd.Timestamp, *, max_age_days: int, max_pool: int) -> pd.DataFrame:
    dates = group[DATE_COL].to_numpy(dtype="datetime64[ns]")
    pos = int(np.searchsorted(dates, np.datetime64(valuation_date), side="left"))
    if pos <= 0:
        return group.iloc[0:0].copy()
    start = max(0, pos - max_pool)
    out = group.iloc[start:pos].copy()
    min_date = pd.Timestamp(valuation_date) - pd.Timedelta(days=int(max_age_days))
    return out[out[DATE_COL] >= min_date].copy()


def _concat_candidates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True)
    if "_row_id" in out.columns:
        out = out.drop_duplicates("_row_id", keep="first")
    return out.reset_index(drop=True)


def _score_candidates(target: pd.Series, candidates: pd.DataFrame, cfg: CompEngineConfig) -> pd.DataFrame:
    out = candidates.copy()
    valuation_date = pd.Timestamp(target[DATE_COL])
    recency = (valuation_date - pd.to_datetime(out[DATE_COL], errors="coerce")).dt.days.astype("float64")
    out["_recency_days"] = recency
    out = out[(out["_recency_days"] > 0) & (out["_recency_days"] <= cfg.fallback_max_age_days)].copy()
    if out.empty:
        return out

    out["_distance_km"] = _distance_to_target_km(target, out)
    sqft_diff = _relative_diff(out["_gross_square_feet"], target.get("_gross_square_feet"))
    age_diff = _absolute_diff(out["_building_age"], target.get("_building_age"))
    total_unit_diff = _absolute_diff(out["_total_units"], target.get("_total_units"))
    residential_unit_diff = _absolute_diff(out["_residential_units"], target.get("_residential_units"))
    unit_diff = pd.concat([total_unit_diff, residential_unit_diff], axis=1).min(axis=1)

    same_segment = out["_segment_key"].astype("string") == str(target.get("_segment_key", ""))
    same_class = out["_building_class_key"].astype("string") == str(target.get("_building_class_key", ""))
    same_h3 = out["_h3_key"].astype("string") == str(target.get("_h3_key", ""))

    distance_component = (out["_distance_km"] / cfg.max_distance_km).clip(lower=0, upper=3).fillna(1.25)
    recency_component = (out["_recency_days"] / cfg.fallback_max_age_days).clip(lower=0, upper=1)
    sqft_component = sqft_diff.clip(lower=0, upper=3).fillna(0.75)
    age_component = (age_diff / cfg.max_age_diff_years).clip(lower=0, upper=3).fillna(0.75)
    unit_component = (unit_diff / cfg.max_unit_diff).clip(lower=0, upper=3).fillna(0.75)
    segment_component = np.where(same_segment, 0.0, 0.5)
    class_component = np.where(same_class, 0.0, 0.15)
    h3_component = np.where(same_h3, 0.0, 0.20)

    out["_comp_score"] = (
        0.30 * distance_component
        + 0.20 * recency_component
        + 0.20 * sqft_component
        + 0.10 * age_component
        + 0.08 * unit_component
        + 0.06 * segment_component
        + 0.03 * class_component
        + 0.03 * h3_component
    ).astype("float64")

    distance_ok = out["_distance_km"].isna() | (out["_distance_km"] <= cfg.max_distance_km)
    sqft_ok = sqft_diff.isna() | (sqft_diff <= cfg.max_sqft_ratio_diff)
    age_ok = age_diff.isna() | (age_diff <= cfg.max_age_diff_years)
    unit_ok = unit_diff.isna() | (unit_diff <= cfg.max_unit_diff)
    primary_recency_ok = out["_recency_days"] <= cfg.primary_max_age_days
    compatible_ok = out["_segment_key"].isin(_compatible_segments(target.get("_segment_key")))
    out["_primary_eligible"] = distance_ok & sqft_ok & age_ok & unit_ok & primary_recency_ok & compatible_ok
    return out


def _with_normalized_weights(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    raw = 1.0 / (0.05 + pd.to_numeric(out["_comp_score"], errors="coerce").clip(lower=0))
    total = float(raw.sum())
    out["_comp_weight"] = raw / total if total > 0 else 1.0 / max(len(out), 1)
    return out


def _aggregate_features(target: pd.Series, selected: pd.DataFrame) -> dict[str, float]:
    if selected.empty:
        return {
            "comp_count": 0.0,
            "comp_median_price": np.nan,
            "comp_median_ppsf": np.nan,
            "comp_weighted_estimate": np.nan,
            "comp_price_dispersion": np.nan,
            "comp_nearest_distance_km": np.nan,
            "comp_median_recency_days": np.nan,
            "comp_local_momentum": np.nan,
        }

    prices = pd.to_numeric(selected[TARGET_COL], errors="coerce")
    ppsf = pd.to_numeric(selected["_comp_ppsf"], errors="coerce")
    weights = pd.to_numeric(selected.get("_comp_weight"), errors="coerce").fillna(0)
    target_sqft = _finite_positive(target.get("_gross_square_feet"))

    median_ppsf = float(ppsf.dropna().median()) if ppsf.notna().any() else np.nan
    weighted_ppsf = _weighted_average(ppsf, weights)
    weighted_price = _weighted_average(prices, weights)
    if target_sqft is not None and np.isfinite(weighted_ppsf):
        weighted_estimate = float(weighted_ppsf * target_sqft)
    else:
        weighted_estimate = float(weighted_price) if np.isfinite(weighted_price) else np.nan

    dispersion_source = ppsf if ppsf.notna().sum() >= 2 else prices
    dispersion = _median_abs_pct_deviation(dispersion_source)

    return {
        "comp_count": float(len(selected)),
        "comp_median_price": float(prices.dropna().median()) if prices.notna().any() else np.nan,
        "comp_median_ppsf": median_ppsf,
        "comp_weighted_estimate": weighted_estimate,
        "comp_price_dispersion": dispersion,
        "comp_nearest_distance_km": float(selected["_distance_km"].min()) if selected["_distance_km"].notna().any() else np.nan,
        "comp_median_recency_days": float(selected["_recency_days"].median()) if selected["_recency_days"].notna().any() else np.nan,
        "comp_local_momentum": _local_momentum(selected),
    }


def _selected_comp_records(
    target: pd.Series,
    selected: pd.DataFrame,
    split_name: str,
    eligibility_scope: str,
) -> list[dict[str, Any]]:
    rows = []
    valuation_date = pd.Timestamp(target[DATE_COL]).strftime("%Y-%m-%d")
    valuation_property_id = _display_value(target.get("property_id"))
    for rank, (_, comp) in enumerate(selected.sort_values("_comp_score").iterrows(), start=1):
        rows.append(
            {
                "valuation_row_id": str(target.get("_row_id", "")),
                "valuation_split": split_name,
                "valuation_sale_date": valuation_date,
                "valuation_property_id": valuation_property_id,
                "comp_rank": int(rank),
                "comp_row_id": str(comp.get("_row_id", "")),
                "comp_property_id": _display_value(comp.get("property_id")),
                "comp_sale_date": pd.Timestamp(comp[DATE_COL]).strftime("%Y-%m-%d"),
                "comp_sale_price": _float_or_nan(comp.get(TARGET_COL)),
                "comp_ppsf": _float_or_nan(comp.get("_comp_ppsf")),
                "comp_borough": _display_value(comp.get("borough")),
                "comp_property_segment": _display_value(comp.get("property_segment")),
                "comp_building_class": _display_value(comp.get("building_class")),
                "comp_h3_index": _display_value(comp.get("h3_index")),
                "comp_distance_km": _float_or_nan(comp.get("_distance_km")),
                "comp_recency_days": _float_or_nan(comp.get("_recency_days")),
                "comp_score": _float_or_nan(comp.get("_comp_score")),
                "comp_weight": _float_or_nan(comp.get("_comp_weight")),
                "eligibility_scope": eligibility_scope,
            }
        )
    return rows


def _frame_manifest(features: pd.DataFrame, selected_rows: list[dict[str, Any]], cfg: CompEngineConfig) -> dict[str, Any]:
    comp_count = pd.to_numeric(features.get("comp_count"), errors="coerce").fillna(0)
    no_comp_rate = float((comp_count <= 0).mean()) if len(comp_count) else 1.0
    sparse_rate = float((comp_count < cfg.min_comps).mean()) if len(comp_count) else 1.0
    return {
        "row_count": int(len(features)),
        "selected_comps_rows": int(len(selected_rows)),
        "comp_count": {
            "min": float(comp_count.min()) if len(comp_count) else 0.0,
            "median": float(comp_count.median()) if len(comp_count) else 0.0,
            "max": float(comp_count.max()) if len(comp_count) else 0.0,
            "no_comp_rate": no_comp_rate,
            "sparse_comp_rate": sparse_rate,
        },
        "missing_rates": {
            feature: float(features[feature].isna().mean()) if feature in features.columns and len(features) else 1.0
            for feature in COMP_FEATURES
        },
    }


def _compatible_segments(segment: Any) -> list[str]:
    key = str(segment or "").strip().upper()
    if not key or key in {"NAN", "NONE", "<NA>"}:
        return [""]
    apartment = {"ELEVATOR", "WALKUP"}
    low_density = {"SINGLE_FAMILY", "SMALL_MULTI"}
    if key in apartment:
        return sorted(apartment)
    if key in low_density:
        return sorted(low_density)
    return [key]


def _string_key(series_or_value: Any) -> Any:
    if isinstance(series_or_value, pd.Series):
        return series_or_value.astype("string").fillna("").str.strip().str.upper()
    if series_or_value is None or pd.isna(series_or_value):
        return ""
    return str(series_or_value).strip().upper()


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _distance_to_target_km(target: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    lat1 = _finite_positive_or_any(target.get("_latitude"))
    lon1 = _finite_positive_or_any(target.get("_longitude"))
    lat2 = pd.to_numeric(candidates.get("_latitude"), errors="coerce")
    lon2 = pd.to_numeric(candidates.get("_longitude"), errors="coerce")
    if lat1 is None or lon1 is None:
        return pd.Series(np.nan, index=candidates.index, dtype="float64")
    lat_diff = (lat2 - lat1) * 111.0
    lon_diff = (lon2 - lon1) * 111.0 * np.cos(np.radians(float(lat1)))
    distance = np.sqrt(lat_diff**2 + lon_diff**2)
    return pd.Series(distance, index=candidates.index, dtype="float64")


def _relative_diff(series: pd.Series, value: Any) -> pd.Series:
    base = _finite_positive(value)
    numeric = pd.to_numeric(series, errors="coerce")
    if base is None:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return (numeric - base).abs() / base


def _absolute_diff(series: pd.Series, value: Any) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        base = float(value)
    except (TypeError, ValueError):
        base = np.nan
    if not np.isfinite(base):
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return (numeric - base).abs()


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = vals.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    total = float(w[mask].sum())
    if total <= 0:
        return float("nan")
    return float((vals[mask] * w[mask]).sum() / total)


def _median_abs_pct_deviation(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < 2:
        return float("nan")
    median = float(vals.median())
    if not np.isfinite(median) or median <= 0:
        return float("nan")
    return float((vals / median - 1.0).abs().median())


def _local_momentum(selected: pd.DataFrame) -> float:
    if len(selected) < 4:
        return float("nan")
    ordered = selected.sort_values(DATE_COL)
    ppsf = pd.to_numeric(ordered["_comp_ppsf"], errors="coerce")
    if ppsf.notna().sum() < 4:
        return float("nan")
    midpoint = len(ordered) // 2
    older = ppsf.iloc[:midpoint].dropna()
    recent = ppsf.iloc[midpoint:].dropna()
    if older.empty or recent.empty:
        return float("nan")
    older_median = float(older.median())
    if older_median <= 0 or not np.isfinite(older_median):
        return float("nan")
    return float(float(recent.median()) / older_median - 1.0)


def _finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0:
        return None
    return out


def _finite_positive_or_any(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _display_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)
