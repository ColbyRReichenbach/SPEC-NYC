"""Model evaluation utilities and CLI for S.P.E.C. NYC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def ppe10(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Percent of predictions within +/-10% absolute percentage error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    valid = y_true_arr > 0
    if valid.sum() == 0:
        return 0.0
    ape = np.abs((y_pred_arr[valid] - y_true_arr[valid]) / y_true_arr[valid])
    return float((ape <= 0.10).mean())


def mdape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Median absolute percentage error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    valid = y_true_arr > 0
    if valid.sum() == 0:
        return float("nan")
    ape = np.abs((y_pred_arr[valid] - y_true_arr[valid]) / y_true_arr[valid])
    return float(np.median(ape))


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    y_true_col: str = "sale_price",
    y_pred_col: str = "predicted_price",
    segment_col: str = "property_segment",
    tier_col: str = "price_tier",
) -> Dict[str, object]:
    """Compute overall and grouped model metrics."""
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError(f"Expected columns '{y_true_col}' and '{y_pred_col}' in prediction dataframe.")

    y_true = pd.to_numeric(df[y_true_col], errors="coerce")
    y_pred = pd.to_numeric(df[y_pred_col], errors="coerce")
    valid = y_true.notna() & y_pred.notna() & (y_true > 0)
    if valid.sum() == 0:
        raise ValueError("No valid rows with numeric y_true/y_pred and y_true > 0 for evaluation.")

    frame = df.loc[valid].copy()
    frame[y_true_col] = y_true.loc[valid]
    frame[y_pred_col] = y_pred.loc[valid]
    frame["abs_pct_error"] = np.abs((frame[y_pred_col] - frame[y_true_col]) / frame[y_true_col])

    overall = {
        "n": int(len(frame)),
        "ppe10": ppe10(frame[y_true_col], frame[y_pred_col]),
        "mdape": mdape(frame[y_true_col], frame[y_pred_col]),
        "r2": float(r2_score(frame[y_true_col], frame[y_pred_col])),
    }

    per_segment = _group_metrics(frame, group_col=segment_col, y_true_col=y_true_col, y_pred_col=y_pred_col)
    per_price_tier = _group_metrics(frame, group_col=tier_col, y_true_col=y_true_col, y_pred_col=y_pred_col)

    return {
        "overall": overall,
        "per_segment": per_segment,
        "per_price_tier": per_price_tier,
    }


def build_segment_scorecard(
    df: pd.DataFrame,
    *,
    y_true_col: str = "sale_price",
    y_pred_col: str = "predicted_price",
    segment_col: str = "property_segment",
    tier_col: str = "price_tier",
) -> pd.DataFrame:
    """Build a flat scorecard table for per-segment and per-tier metrics."""
    by_segment = _group_metrics_table(df, segment_col, y_true_col=y_true_col, y_pred_col=y_pred_col, group_type="segment")
    by_tier = _group_metrics_table(df, tier_col, y_true_col=y_true_col, y_pred_col=y_pred_col, group_type="price_tier")
    return pd.concat([by_segment, by_tier], ignore_index=True)


def _group_metrics(
    frame: pd.DataFrame,
    *,
    group_col: str,
    y_true_col: str,
    y_pred_col: str,
) -> Dict[str, Dict[str, float]]:
    if group_col not in frame.columns:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for group, group_df in frame.groupby(group_col):
        if len(group_df) < 2:
            continue
        label = str(group)
        out[label] = {
            "n": int(len(group_df)),
            "ppe10": ppe10(group_df[y_true_col], group_df[y_pred_col]),
            "mdape": mdape(group_df[y_true_col], group_df[y_pred_col]),
            "r2": float(r2_score(group_df[y_true_col], group_df[y_pred_col])),
        }
    return out


def _group_metrics_table(
    frame: pd.DataFrame,
    group_col: str,
    *,
    y_true_col: str,
    y_pred_col: str,
    group_type: str,
) -> pd.DataFrame:
    if group_col not in frame.columns:
        return pd.DataFrame(columns=["group_type", "group_name", "n", "ppe10", "mdape", "r2"])

    records = []
    for group, group_df in frame.groupby(group_col):
        if len(group_df) < 2:
            continue
        records.append(
            {
                "group_type": group_type,
                "group_name": str(group),
                "n": int(len(group_df)),
                "ppe10": ppe10(group_df[y_true_col], group_df[y_pred_col]),
                "mdape": mdape(group_df[y_true_col], group_df[y_pred_col]),
                "r2": float(r2_score(group_df[y_true_col], group_df[y_pred_col])),
            }
        )
    return pd.DataFrame.from_records(records)


def save_metrics(metrics: Dict[str, object], path: Path) -> None:
    """Persist metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model predictions for S.P.E.C. NYC")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="CSV with sale_price + predicted_price")
    parser.add_argument("--output-json", type=Path, default=Path("models/metrics_v1.json"))
    parser.add_argument("--segment-scorecard-csv", type=Path, default=Path("reports/model/segment_scorecard_v1.csv"))
    parser.add_argument("--y-true-col", type=str, default="sale_price")
    parser.add_argument("--y-pred-col", type=str, default="predicted_price")
    parser.add_argument("--segment-col", type=str, default="property_segment")
    parser.add_argument("--tier-col", type=str, default="price_tier")
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions_csv)
    metrics = evaluate_predictions(
        frame,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        segment_col=args.segment_col,
        tier_col=args.tier_col,
    )
    save_metrics(metrics, args.output_json)

    scorecard = build_segment_scorecard(
        frame,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        segment_col=args.segment_col,
        tier_col=args.tier_col,
    )
    args.segment_scorecard_csv.parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(args.segment_scorecard_csv, index=False)

    print(f"Saved metrics JSON: {args.output_json}")
    print(f"Saved segment scorecard: {args.segment_scorecard_csv}")
    print(
        f"Overall => n={metrics['overall']['n']}, PPE10={metrics['overall']['ppe10']:.3f}, "
        f"MdAPE={metrics['overall']['mdape']:.3f}, R2={metrics['overall']['r2']:.3f}"
    )


if __name__ == "__main__":
    _cli()

