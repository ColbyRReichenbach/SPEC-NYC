"""Feature drift monitor for S.P.E.C. NYC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "gross_square_feet",
    "year_built",
    "building_age",
    "residential_units",
    "total_units",
    "distance_to_center_km",
    "h3_price_lag",
]


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index for numeric distributions."""
    exp = pd.to_numeric(expected, errors="coerce").dropna()
    cur = pd.to_numeric(actual, errors="coerce").dropna()
    if len(exp) < 20 or len(cur) < 20:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(exp, quantiles))
    if len(cut_points) < 3:
        return 0.0
    exp_hist, _ = np.histogram(exp, bins=cut_points)
    cur_hist, _ = np.histogram(cur, bins=cut_points)

    exp_pct = np.clip(exp_hist / max(exp_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    psi = np.sum((cur_pct - exp_pct) * np.log(cur_pct / exp_pct))
    return float(psi)


def calculate_ks(expected: pd.Series, actual: pd.Series) -> float:
    """Two-sample KS statistic without scipy dependency."""
    exp = np.sort(pd.to_numeric(expected, errors="coerce").dropna().values)
    cur = np.sort(pd.to_numeric(actual, errors="coerce").dropna().values)
    if len(exp) < 20 or len(cur) < 20:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([exp, cur])))
    exp_cdf = np.searchsorted(exp, values, side="right") / len(exp)
    cur_cdf = np.searchsorted(cur, values, side="right") / len(cur)
    return float(np.max(np.abs(exp_cdf - cur_cdf)))


def monitor_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    features: Iterable[str],
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
    ks_warn: float = 0.10,
    ks_alert: float = 0.20,
) -> pd.DataFrame:
    """Compute PSI/KS drift table for selected features."""
    rows = []
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        psi = calculate_psi(reference_df[feature], current_df[feature])
        ks = calculate_ks(reference_df[feature], current_df[feature])
        status = "ok"
        if (not np.isnan(psi) and psi >= psi_alert) or (not np.isnan(ks) and ks >= ks_alert):
            status = "alert"
        elif (not np.isnan(psi) and psi >= psi_warn) or (not np.isnan(ks) and ks >= ks_warn):
            status = "warn"
        rows.append(
            {
                "feature": feature,
                "reference_n": int(pd.to_numeric(reference_df[feature], errors="coerce").notna().sum()),
                "current_n": int(pd.to_numeric(current_df[feature], errors="coerce").notna().sum()),
                "psi": psi,
                "ks": ks,
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "psi", "ks"], ascending=[True, False, False])


def _status_summary(drift_df: pd.DataFrame) -> str:
    if drift_df.empty:
        return "warn"
    if (drift_df["status"] == "alert").any():
        return "alert"
    if (drift_df["status"] == "warn").any():
        return "warn"
    return "ok"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No drift rows generated."
    return df.to_string(index=False)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Monitor feature drift with PSI and KS.")
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--current-csv", type=Path, required=True)
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--psi-warn", type=float, default=0.10)
    parser.add_argument("--psi-alert", type=float, default=0.25)
    parser.add_argument("--ks-warn", type=float, default=0.10)
    parser.add_argument("--ks-alert", type=float, default=0.20)
    parser.add_argument("--output-csv", type=Path, default=Path("reports/monitoring/drift_latest.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/monitoring/drift_latest.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/monitoring/drift_latest.md"))
    args = parser.parse_args()

    reference_df = pd.read_csv(args.reference_csv, low_memory=False)
    current_df = pd.read_csv(args.current_csv, low_memory=False)
    features: List[str] = [f.strip() for f in args.features.split(",") if f.strip()]

    drift_df = monitor_drift(
        reference_df,
        current_df,
        features=features,
        psi_warn=args.psi_warn,
        psi_alert=args.psi_alert,
        ks_warn=args.ks_warn,
        ks_alert=args.ks_alert,
    )
    summary = {
        "status": _status_summary(drift_df),
        "rows": int(len(drift_df)),
        "alerts": int((drift_df["status"] == "alert").sum()) if not drift_df.empty else 0,
        "warnings": int((drift_df["status"] == "warn").sum()) if not drift_df.empty else 0,
        "reference_csv": str(args.reference_csv),
        "current_csv": str(args.current_csv),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    drift_df.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_md.write_text(
        "\n".join(
            [
                "# Drift Monitor",
                "",
                f"- Status: `{summary['status']}`",
                f"- Alerts: `{summary['alerts']}`",
                f"- Warnings: `{summary['warnings']}`",
                "",
                "```text",
                _markdown_table(drift_df),
                "```",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"summary": summary, "output_csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    _cli()

