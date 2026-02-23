"""Non-leaky price tier proxy construction for routed models.

Proxy tiers are derived only from inference-available signals and explicitly
exclude target columns (e.g., sale_price, price_tier).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

TIER_LABELS = ("entry", "core", "premium", "luxury")

# Coarse location prior; intentionally simple and inference-available.
BOROUGH_PRIOR = {
    "1": 0.35,  # Manhattan
    "2": -0.12,  # Bronx
    "3": 0.08,  # Brooklyn
    "4": 0.04,  # Queens
    "5": -0.15,  # Staten Island
}


def compute_price_tier_proxy_score(df: pd.DataFrame) -> pd.Series:
    """Compute a non-target proxy score from inference-available inputs only."""
    idx = df.index
    gross_sqft = _num(df, "gross_square_feet")
    building_age = _num(df, "building_age")
    distance_km = _num(df, "distance_to_center_km")
    total_units = _num(df, "total_units")
    residential_units = _num(df, "residential_units")

    # Stabilize raw magnitudes using monotonic transforms.
    size_signal = np.log1p(gross_sqft.clip(lower=0))
    age_signal = -np.log1p(building_age.clip(lower=0))
    center_signal = -np.log1p(distance_km.clip(lower=0))
    unit_signal = np.log1p(np.maximum(total_units.fillna(0), residential_units.fillna(0)).clip(lower=0))

    borough = _borough_token(df)
    borough_signal = borough.map(BOROUGH_PRIOR).fillna(0.0).astype(float)

    score = (
        0.45 * size_signal
        + 0.30 * center_signal
        + 0.15 * age_signal
        + 0.05 * unit_signal
        + 0.05 * borough_signal
    )
    return pd.Series(score, index=idx, dtype="float64")


def fit_price_tier_proxy_bins(
    df: pd.DataFrame,
    *,
    segment_col: str = "property_segment",
    min_segment_rows: int = 800,
) -> Dict[str, Any]:
    """Fit per-segment and global proxy score thresholds (q25/q50/q75)."""
    work = pd.DataFrame(
        {
            "segment": _segment_token(df, segment_col),
            "score": compute_price_tier_proxy_score(df),
        },
        index=df.index,
    )
    global_bins = _quantile_bins(work["score"])
    bins: Dict[str, Any] = {
        "version": 1,
        "segment_col": segment_col,
        "min_segment_rows": int(min_segment_rows),
        "global": global_bins,
        "segments": {},
    }
    for segment, gdf in work.groupby("segment"):
        valid = gdf["score"].dropna()
        if len(valid) < min_segment_rows:
            continue
        bins["segments"][str(segment)] = {
            **_quantile_bins(valid),
            "n": int(len(valid)),
        }
    return bins


def assign_price_tier_proxy(
    df: pd.DataFrame,
    *,
    bins: Dict[str, Any] | None = None,
    segment_col: str = "property_segment",
    min_segment_rows: int = 800,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign price_tier_proxy with explicit sparse/unknown fallbacks."""
    out = df.copy()
    score = compute_price_tier_proxy_score(out)
    resolved_bins = bins or fit_price_tier_proxy_bins(
        out,
        segment_col=segment_col,
        min_segment_rows=min_segment_rows,
    )

    segment_tokens = _segment_token(out, segment_col)
    global_thresholds = resolved_bins.get("global") or _quantile_bins(score)
    tiers = _bucketize(score, global_thresholds)
    source = pd.Series("global_fallback", index=out.index, dtype="string")

    segment_bins = resolved_bins.get("segments", {})
    for segment, thresholds in segment_bins.items():
        mask = segment_tokens == str(segment)
        if not bool(mask.any()):
            continue
        tiers.loc[mask] = _bucketize(score.loc[mask], thresholds)
        source.loc[mask] = f"segment:{segment}"

    # Explicit fallback for rows missing the proxy score signal bundle.
    missing_score = score.isna()
    tiers.loc[missing_score] = "core"
    source.loc[missing_score] = "default_core_missing_score"

    out["price_tier_proxy_score"] = score
    out["price_tier_proxy"] = pd.Categorical(tiers, categories=list(TIER_LABELS), ordered=True)
    out["price_tier_proxy_source"] = source
    return out, resolved_bins


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _segment_token(df: pd.DataFrame, segment_col: str) -> pd.Series:
    if segment_col not in df.columns:
        return pd.Series("__missing__", index=df.index, dtype="string")
    token = df[segment_col].astype("string").fillna("__missing__").str.strip()
    token = token.replace("", "__missing__")
    return token.astype("string")


def _borough_token(df: pd.DataFrame) -> pd.Series:
    if "borough" not in df.columns:
        return pd.Series("__missing__", index=df.index, dtype="string")
    text = df["borough"].astype("string").fillna("__missing__")
    # Handles numeric borough codes and already-stringed values.
    extracted = text.str.extract(r"(\d+)")[0].fillna(text)
    return extracted.astype("string")


def _quantile_bins(series: pd.Series) -> Dict[str, float]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return {"q25": 0.0, "q50": 0.0, "q75": 0.0, "n": 0}
    q = valid.quantile([0.25, 0.50, 0.75])
    return {
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]),
        "n": int(len(valid)),
    }


def _bucketize(scores: pd.Series, thresholds: Dict[str, Any]) -> pd.Series:
    q25 = float(thresholds.get("q25", 0.0))
    q50 = float(thresholds.get("q50", q25))
    q75 = float(thresholds.get("q75", q50))
    values = np.select(
        [
            scores <= q25,
            scores <= q50,
            scores <= q75,
        ],
        [
            "entry",
            "core",
            "premium",
        ],
        default="luxury",
    )
    out = pd.Series(values, index=scores.index, dtype="string")
    out.loc[scores.isna()] = "core"
    return out
