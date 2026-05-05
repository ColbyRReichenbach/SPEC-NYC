"""AVM-specific valuation diagnostics.

These metrics are intentionally separate from generic regression metrics because
AVM review needs ratio, dispersion, and over/under valuation behavior.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def ppe(y_true: Iterable[float], y_pred: Iterable[float], threshold: float) -> float:
    """Percent of predictions within +/- threshold absolute percentage error."""
    y_true_arr, y_pred_arr = _valid_arrays(y_true, y_pred)
    if len(y_true_arr) == 0:
        return 0.0
    ape = np.abs((y_pred_arr - y_true_arr) / y_true_arr)
    return float((ape <= threshold).mean())


def mdape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Median absolute percentage error."""
    y_true_arr, y_pred_arr = _valid_arrays(y_true, y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    ape = np.abs((y_pred_arr - y_true_arr) / y_true_arr)
    return float(np.median(ape))


def mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean absolute percentage error. Use as secondary evidence due outlier sensitivity."""
    y_true_arr, y_pred_arr = _valid_arrays(y_true, y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    ape = np.abs((y_pred_arr - y_true_arr) / y_true_arr)
    return float(np.mean(ape))


def valuation_ratios(y_true: Iterable[float], y_pred: Iterable[float]) -> np.ndarray:
    """Predicted value divided by observed sale price."""
    y_true_arr, y_pred_arr = _valid_arrays(y_true, y_pred)
    if len(y_true_arr) == 0:
        return np.asarray([], dtype=float)
    return y_pred_arr / y_true_arr


def coefficient_of_dispersion(ratios: Iterable[float]) -> float:
    """IAAO-style COD: average absolute deviation from median ratio / median ratio * 100."""
    ratio_arr = _valid_positive_array(ratios)
    if len(ratio_arr) == 0:
        return float("nan")
    median_ratio = float(np.median(ratio_arr))
    if median_ratio <= 0:
        return float("nan")
    return float(np.mean(np.abs(ratio_arr - median_ratio)) / median_ratio * 100.0)


def price_related_differential(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """PRD: mean valuation ratio divided by weighted mean valuation ratio."""
    y_true_arr, y_pred_arr = _valid_arrays(y_true, y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    mean_ratio = float(np.mean(y_pred_arr / y_true_arr))
    weighted_mean_ratio = float(np.sum(y_pred_arr) / np.sum(y_true_arr))
    if weighted_mean_ratio == 0:
        return float("nan")
    return float(mean_ratio / weighted_mean_ratio)


def price_related_bias(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """
    PRB-style slope: relative ratio deviation regressed on log2 sale-price scale.

    Positive values indicate ratios tend to rise with price; negative values indicate
    ratios tend to fall with price. This is a diagnostic, not a formal appraisal
    compliance conclusion.
    """
    y_true_arr, y_pred_arr = _valid_arrays(y_true, y_pred)
    if len(y_true_arr) < 3:
        return float("nan")
    ratios = y_pred_arr / y_true_arr
    median_ratio = float(np.median(ratios))
    median_price = float(np.median(y_true_arr))
    if median_ratio <= 0 or median_price <= 0:
        return float("nan")
    x = np.log2(y_true_arr / median_price)
    if len(np.unique(x)) < 2:
        return float("nan")
    y = (ratios - median_ratio) / median_ratio
    return float(np.polyfit(x, y, 1)[0])


def avm_metric_summary(
    df: pd.DataFrame,
    *,
    y_true_col: str = "sale_price",
    y_pred_col: str = "predicted_price",
    interval_lower_col: str | None = None,
    interval_upper_col: str | None = None,
    hit_status_col: str | None = None,
) -> dict[str, float | int | None]:
    """Compute AVM metrics for one frame or slice."""
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError(f"Expected columns '{y_true_col}' and '{y_pred_col}' in prediction dataframe.")

    frame = _valid_prediction_frame(df, y_true_col=y_true_col, y_pred_col=y_pred_col)
    if frame.empty:
        raise ValueError("No valid rows with numeric y_true/y_pred and y_true > 0 for evaluation.")

    y_true = frame[y_true_col].to_numpy(dtype=float)
    y_pred = frame[y_pred_col].to_numpy(dtype=float)
    signed_pct_error = (y_pred - y_true) / y_true
    ratios = y_pred / y_true

    metrics: dict[str, float | int | None] = {
        "n": int(len(frame)),
        "ppe5": _json_number(ppe(y_true, y_pred, 0.05)),
        "ppe10": _json_number(ppe(y_true, y_pred, 0.10)),
        "ppe20": _json_number(ppe(y_true, y_pred, 0.20)),
        "mdape": _json_number(mdape(y_true, y_pred)),
        "mape": _json_number(mape(y_true, y_pred)),
        "r2": _json_number(float(r2_score(y_true, y_pred))) if len(frame) >= 2 else None,
        "median_valuation_ratio": _json_number(float(np.median(ratios))),
        "mean_valuation_ratio": _json_number(float(np.mean(ratios))),
        "coefficient_of_dispersion": _json_number(coefficient_of_dispersion(ratios)),
        "price_related_differential": _json_number(price_related_differential(y_true, y_pred)),
        "price_related_bias": _json_number(price_related_bias(y_true, y_pred)),
        "signed_pct_error_mean": _json_number(float(np.mean(signed_pct_error))),
        "signed_pct_error_median": _json_number(float(np.median(signed_pct_error))),
        "overvaluation_rate_10": _json_number(float((signed_pct_error > 0.10).mean())),
        "undervaluation_rate_10": _json_number(float((signed_pct_error < -0.10).mean())),
        "overvaluation_rate_20": _json_number(float((signed_pct_error > 0.20).mean())),
        "undervaluation_rate_20": _json_number(float((signed_pct_error < -0.20).mean())),
    }

    lower = interval_lower_col or _first_existing_column(
        frame,
        ["prediction_interval_lower", "interval_lower", "lower_bound", "p50_lower"],
    )
    upper = interval_upper_col or _first_existing_column(
        frame,
        ["prediction_interval_upper", "interval_upper", "upper_bound", "p50_upper"],
    )
    if lower and upper and lower in frame.columns and upper in frame.columns:
        low = pd.to_numeric(frame[lower], errors="coerce")
        high = pd.to_numeric(frame[upper], errors="coerce")
        interval_valid = low.notna() & high.notna()
        if bool(interval_valid.any()):
            covered = (frame.loc[interval_valid, y_true_col] >= low.loc[interval_valid]) & (
                frame.loc[interval_valid, y_true_col] <= high.loc[interval_valid]
            )
            metrics["interval_coverage"] = _json_number(float(covered.mean()))
            metrics["interval_rows"] = int(interval_valid.sum())

    hit_col = hit_status_col or ("hit_status" if "hit_status" in frame.columns else None)
    if hit_col and hit_col in frame.columns:
        statuses = frame[hit_col].astype("string").str.lower()
        metrics["hit_rate"] = _json_number(float((statuses == "hit").mean()))
        metrics["low_confidence_hit_rate"] = _json_number(float((statuses == "low_confidence_hit").mean()))
        metrics["no_hit_rate"] = _json_number(float((statuses == "no_hit").mean()))

    return metrics


def avm_scorecard(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_type: str,
    y_true_col: str = "sale_price",
    y_pred_col: str = "predicted_price",
) -> pd.DataFrame:
    """Build a flat AVM metric scorecard for one group column."""
    if group_col not in df.columns:
        return pd.DataFrame(columns=SCORECARD_COLUMNS)

    records = []
    for group, group_df in df.groupby(group_col, dropna=False):
        if len(group_df) < 2:
            continue
        metrics = avm_metric_summary(group_df, y_true_col=y_true_col, y_pred_col=y_pred_col)
        records.append({"group_type": group_type, "group_name": str(group), **metrics})
    if not records:
        return pd.DataFrame(columns=SCORECARD_COLUMNS)
    return pd.DataFrame.from_records(records)


SCORECARD_COLUMNS = [
    "group_type",
    "group_name",
    "n",
    "ppe5",
    "ppe10",
    "ppe20",
    "mdape",
    "mape",
    "r2",
    "median_valuation_ratio",
    "mean_valuation_ratio",
    "coefficient_of_dispersion",
    "price_related_differential",
    "price_related_bias",
    "signed_pct_error_mean",
    "signed_pct_error_median",
    "overvaluation_rate_10",
    "undervaluation_rate_10",
    "overvaluation_rate_20",
    "undervaluation_rate_20",
]


def _valid_prediction_frame(df: pd.DataFrame, *, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    y_true = pd.to_numeric(df[y_true_col], errors="coerce")
    y_pred = pd.to_numeric(df[y_pred_col], errors="coerce")
    valid = y_true.notna() & y_pred.notna() & (y_true > 0)
    frame = df.loc[valid].copy()
    frame[y_true_col] = y_true.loc[valid]
    frame[y_pred_col] = y_pred.loc[valid]
    return frame


def _valid_arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr) & (y_true_arr > 0)
    return y_true_arr[valid], y_pred_arr[valid]


def _valid_positive_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def _json_number(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _first_existing_column(df: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    return None
