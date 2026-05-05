"""Governed real-estate EDA artifacts for the AVM workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/processed/nyc_sales_2019_2024_avm_training.csv")
DEFAULT_OUTPUT_DIR = Path("reports/eda")
IMPORTANT_COLUMNS = [
    "sale_date",
    "sale_price",
    "price_per_sqft",
    "borough",
    "neighborhood",
    "property_segment",
    "building_class",
    "gross_square_feet",
    "year_built",
    "building_age",
    "total_units",
    "residential_units",
    "distance_to_center_km",
    "h3_index",
    "property_id",
    "sqft_imputed",
    "year_built_imputed",
]
SIGNAL_FEATURES = [
    "distance_to_center_km",
    "gross_square_feet",
    "building_age",
    "total_units",
    "residential_units",
]


@dataclass(frozen=True)
class EdaArtifactResult:
    output_dir: Path
    report_path: Path
    manifest_path: Path
    data_profile_path: Path
    segment_region_summary_path: Path
    quarterly_trends_path: Path
    feature_interactions_path: Path
    hypothesis_backlog_path: Path
    model_error_slices_path: Path | None


def run_eda(
    *,
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tag: str | None = None,
    predictions_csv: Path | None = None,
    limit: int | None = None,
) -> EdaArtifactResult:
    """Run the senior-DS EDA workflow and persist audit artifacts."""
    run_tag = _run_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    sales = load_sales_frame(input_csv, limit=limit)
    profile = build_data_profile(sales, source_path=input_csv)
    segment_region = build_segment_region_summary(sales)
    trends = build_quarterly_market_trends(sales)
    interactions = build_feature_interaction_signals(sales)

    error_slices = None
    if predictions_csv is not None and predictions_csv.exists():
        predictions = pd.read_csv(predictions_csv, low_memory=False)
        error_slices = build_model_error_slices(predictions)

    hypotheses = build_hypothesis_backlog(
        segment_region=segment_region,
        interactions=interactions,
        error_slices=error_slices,
    )

    paths = {
        "data_profile": output_dir / f"data_profile_{run_tag}.json",
        "segment_region_summary": output_dir / f"segment_region_summary_{run_tag}.csv",
        "quarterly_market_trends": output_dir / f"quarterly_market_trends_{run_tag}.csv",
        "feature_interaction_signals": output_dir / f"feature_interaction_signals_{run_tag}.csv",
        "hypothesis_backlog": output_dir / f"hypothesis_backlog_{run_tag}.md",
        "report": output_dir / f"avm_eda_report_{run_tag}.md",
        "manifest": output_dir / f"eda_manifest_{run_tag}.json",
    }
    model_error_path = output_dir / f"model_error_slices_{run_tag}.csv" if error_slices is not None else None

    _write_json(paths["data_profile"], profile)
    segment_region.to_csv(paths["segment_region_summary"], index=False)
    trends.to_csv(paths["quarterly_market_trends"], index=False)
    interactions.to_csv(paths["feature_interaction_signals"], index=False)
    paths["hypothesis_backlog"].write_text(hypotheses, encoding="utf-8")
    if error_slices is not None and model_error_path is not None:
        error_slices.to_csv(model_error_path, index=False)

    report = build_eda_report(
        profile=profile,
        segment_region=segment_region,
        trends=trends,
        interactions=interactions,
        error_slices=error_slices,
        hypotheses=hypotheses,
        predictions_csv=predictions_csv,
    )
    paths["report"].write_text(report, encoding="utf-8")

    manifest = {
        "run_tag": run_tag,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "command": " ".join(sys.argv),
        "input_csv": str(input_csv),
        "predictions_csv": str(predictions_csv) if predictions_csv else None,
        "limit": limit,
        "artifacts": {
            "report": str(paths["report"]),
            "data_profile": str(paths["data_profile"]),
            "segment_region_summary": str(paths["segment_region_summary"]),
            "quarterly_market_trends": str(paths["quarterly_market_trends"]),
            "feature_interaction_signals": str(paths["feature_interaction_signals"]),
            "hypothesis_backlog": str(paths["hypothesis_backlog"]),
            "model_error_slices": str(model_error_path) if model_error_path else None,
        },
        "status": "complete",
        "workflow_stage": "eda_before_hypothesis_queue",
        "external_references": [
            "https://researchexchange.iaao.org/jptaa/vol15/iss2/5/",
            "https://zillow.zendesk.com/hc/en-us/articles/4402325964563-How-is-the-Zestimate-calculated",
            "https://www.fanniemae.com/research-and-insights/perspectives/advancing-collateral-valuation",
        ],
    }
    _write_json(paths["manifest"], manifest)

    return EdaArtifactResult(
        output_dir=output_dir,
        report_path=paths["report"],
        manifest_path=paths["manifest"],
        data_profile_path=paths["data_profile"],
        segment_region_summary_path=paths["segment_region_summary"],
        quarterly_trends_path=paths["quarterly_market_trends"],
        feature_interactions_path=paths["feature_interaction_signals"],
        hypothesis_backlog_path=paths["hypothesis_backlog"],
        model_error_slices_path=model_error_path,
    )


def load_sales_frame(input_csv: Path, *, limit: int | None = None) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"EDA input CSV not found: {input_csv}")
    usecols = _available_usecols(input_csv, IMPORTANT_COLUMNS)
    frame = pd.read_csv(input_csv, usecols=usecols or None, low_memory=False)
    if limit is not None:
        frame = frame.head(limit).copy()
    return prepare_sales_frame(frame)


def prepare_sales_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    if "sale_date" in out.columns:
        out["sale_date"] = pd.to_datetime(out["sale_date"], errors="coerce")
    for column in [
        "sale_price",
        "price_per_sqft",
        "gross_square_feet",
        "year_built",
        "building_age",
        "total_units",
        "residential_units",
        "distance_to_center_km",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "price_per_sqft" not in out.columns and {"sale_price", "gross_square_feet"}.issubset(out.columns):
        sqft = out["gross_square_feet"].where(out["gross_square_feet"] > 0)
        out["price_per_sqft"] = out["sale_price"] / sqft
    if "sale_price" in out.columns:
        out = out[out["sale_price"].notna() & (out["sale_price"] > 0)].copy()
    return out


def build_data_profile(frame: pd.DataFrame, *, source_path: Path | None = None) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "source_path": str(source_path) if source_path else None,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "sale_date_min": _date_bound(frame, "sale_date", "min"),
        "sale_date_max": _date_bound(frame, "sale_date", "max"),
        "sale_price": _numeric_summary(frame, "sale_price"),
        "price_per_sqft": _numeric_summary(frame, "price_per_sqft"),
        "important_missingness": {
            column: float(frame[column].isna().mean())
            for column in IMPORTANT_COLUMNS
            if column in frame.columns and len(frame) > 0
        },
        "cardinality": {
            column: int(frame[column].nunique(dropna=True))
            for column in ["borough", "neighborhood", "property_segment", "building_class", "h3_index", "property_id"]
            if column in frame.columns
        },
    }
    for flag in ["sqft_imputed", "year_built_imputed"]:
        if flag in frame.columns:
            profile[f"{flag}_rate"] = float(_truthy(frame[flag]).mean())
    return profile


def build_segment_region_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"borough", "property_segment", "sale_price", "price_per_sqft"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=_segment_columns())
    rows = []
    grouped = frame.groupby(["borough", "property_segment"], dropna=False)
    total_rows = max(len(frame), 1)
    for (borough, segment), group in grouped:
        prices = pd.to_numeric(group["sale_price"], errors="coerce")
        ppsf = pd.to_numeric(group["price_per_sqft"], errors="coerce")
        rows.append(
            {
                "borough": str(borough),
                "property_segment": str(segment),
                "n": int(len(group)),
                "row_share": float(len(group) / total_rows),
                "median_sale_price": _median(prices),
                "median_ppsf": _median(ppsf),
                "p25_ppsf": _quantile(ppsf, 0.25),
                "p75_ppsf": _quantile(ppsf, 0.75),
                "ppsf_iqr": _quantile(ppsf, 0.75) - _quantile(ppsf, 0.25),
                "median_sqft": _median(group.get("gross_square_feet")),
                "median_building_age": _median(group.get("building_age")),
                "median_distance_to_center_km": _median(group.get("distance_to_center_km")),
                "sqft_imputed_rate": _truthy(group["sqft_imputed"]).mean() if "sqft_imputed" in group.columns else np.nan,
                "year_built_imputed_rate": _truthy(group["year_built_imputed"]).mean() if "year_built_imputed" in group.columns else np.nan,
            }
        )
    out = pd.DataFrame(rows, columns=_segment_columns())
    return out.sort_values(["n", "borough", "property_segment"], ascending=[False, True, True]).reset_index(drop=True)


def build_quarterly_market_trends(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"sale_date", "borough", "property_segment", "sale_price", "price_per_sqft"}.issubset(frame.columns):
        return pd.DataFrame(columns=["period", "borough", "property_segment", "n", "median_sale_price", "median_ppsf", "ppsf_qoq_change"])
    work = frame.dropna(subset=["sale_date"]).copy()
    work["period"] = work["sale_date"].dt.to_period("Q").astype(str)
    rows = []
    for (period, borough, segment), group in work.groupby(["period", "borough", "property_segment"], dropna=False):
        rows.append(
            {
                "period": str(period),
                "borough": str(borough),
                "property_segment": str(segment),
                "n": int(len(group)),
                "median_sale_price": _median(group["sale_price"]),
                "median_ppsf": _median(group["price_per_sqft"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["period", "borough", "property_segment", "n", "median_sale_price", "median_ppsf", "ppsf_qoq_change"])
    out = out.sort_values(["borough", "property_segment", "period"]).reset_index(drop=True)
    out["ppsf_qoq_change"] = out.groupby(["borough", "property_segment"])["median_ppsf"].pct_change()
    return out


def build_feature_interaction_signals(frame: pd.DataFrame, *, min_rows: int = 250) -> pd.DataFrame:
    columns = ["scope", "borough", "property_segment", "feature", "n", "spearman_corr_log_ppsf", "direction", "abs_corr"]
    if "price_per_sqft" not in frame.columns:
        return pd.DataFrame(columns=columns)
    work = frame.copy()
    work = work[pd.to_numeric(work["price_per_sqft"], errors="coerce") > 0].copy()
    work["log_ppsf"] = np.log1p(work["price_per_sqft"])
    rows = []

    scopes: list[tuple[str, str, str, pd.DataFrame]] = [("global", "ALL", "ALL", work)]
    if {"borough", "property_segment"}.issubset(work.columns):
        for (borough, segment), group in work.groupby(["borough", "property_segment"], dropna=False):
            scopes.append(("borough_segment", str(borough), str(segment), group))
    if "borough" in work.columns:
        for borough, group in work.groupby("borough", dropna=False):
            scopes.append(("borough", str(borough), "ALL", group))
    if "property_segment" in work.columns:
        for segment, group in work.groupby("property_segment", dropna=False):
            scopes.append(("segment", "ALL", str(segment), group))

    for scope, borough, segment, group in scopes:
        if len(group) < min_rows:
            continue
        for feature in SIGNAL_FEATURES:
            if feature not in group.columns:
                continue
            corr = _spearman_corr(group[feature], group["log_ppsf"])
            if np.isnan(corr):
                continue
            rows.append(
                {
                    "scope": scope,
                    "borough": borough,
                    "property_segment": segment,
                    "feature": feature,
                    "n": int(len(group[[feature, "log_ppsf"]].dropna())),
                    "spearman_corr_log_ppsf": float(corr),
                    "direction": "positive" if corr > 0.03 else "negative" if corr < -0.03 else "flat",
                    "abs_corr": abs(float(corr)),
                }
            )
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.sort_values(["abs_corr", "scope"], ascending=[False, True]).reset_index(drop=True)


def build_model_error_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if not {"sale_price", "predicted_price"}.issubset(frame.columns):
        return pd.DataFrame(columns=_error_columns())
    frame["sale_price"] = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame["predicted_price"] = pd.to_numeric(frame["predicted_price"], errors="coerce")
    frame = frame[frame["sale_price"].notna() & frame["predicted_price"].notna() & (frame["sale_price"] > 0)].copy()
    if frame.empty:
        return pd.DataFrame(columns=_error_columns())
    frame["abs_pct_error"] = ((frame["predicted_price"] - frame["sale_price"]) / frame["sale_price"]).abs()
    frame["signed_pct_error"] = (frame["predicted_price"] - frame["sale_price"]) / frame["sale_price"]
    frame["valuation_ratio"] = frame["predicted_price"] / frame["sale_price"]

    slice_specs = [("all", None)]
    for col in ["borough", "property_segment", "neighborhood", "price_tier", "model_route"]:
        if col in frame.columns:
            slice_specs.append((col, col))
    if {"borough", "property_segment"}.issubset(frame.columns):
        frame["borough_segment"] = frame["borough"].astype(str) + "|" + frame["property_segment"].astype(str)
        slice_specs.append(("borough_segment", "borough_segment"))
    if "comp_count" in frame.columns:
        frame["comp_count_bucket"] = pd.cut(
            pd.to_numeric(frame["comp_count"], errors="coerce"),
            bins=[-0.1, 0.9, 2.9, 5.9, np.inf],
            labels=["0", "1-2", "3-5", "6+"],
        ).astype("string")
        slice_specs.append(("comp_count_bucket", "comp_count_bucket"))

    rows = []
    for slice_type, column in slice_specs:
        groups = [("ALL", frame)] if column is None else frame.groupby(column, dropna=False)
        for label, group in groups:
            if len(group) < 5:
                continue
            rows.append(_error_row(slice_type, str(label), group))
    out = pd.DataFrame(rows, columns=_error_columns())
    if out.empty:
        return out
    return out.sort_values(["mdape", "n"], ascending=[False, False]).reset_index(drop=True)


def build_hypothesis_backlog(
    *,
    segment_region: pd.DataFrame,
    interactions: pd.DataFrame,
    error_slices: pd.DataFrame | None,
) -> str:
    lines = [
        "# AVM EDA Hypothesis Backlog",
        "",
        "These are candidate hypotheses for the governed experiment UI. Each should be converted into a locked spec before training.",
        "",
    ]
    if error_slices is not None and not error_slices.empty:
        lines.extend(["## Underperforming Slices", ""])
        for _, row in error_slices.head(8).iterrows():
            lines.append(
                f"- Investigate `{row['slice_type']}={row['slice_name']}`: n={int(row['n'])}, "
                f"MdAPE={row['mdape']:.3f}, PPE10={row['ppe10']:.3f}. Candidate action: segment-specific residual calibration or abstention rule."
            )
        lines.append("")
    if not interactions.empty:
        flips = _feature_direction_flips(interactions)
        if flips:
            lines.extend(["## Non-Stationary Feature Effects", ""])
            for feature, details in flips[:8]:
                lines.append(
                    f"- Test interaction or segmented treatment for `{feature}` because observed direction changes across {details}."
                )
            lines.append("")
    if not segment_region.empty:
        sparse = segment_region[segment_region["n"] < 500].head(8)
        if not sparse.empty:
            lines.extend(["## Sparse Segment/Region Cells", ""])
            for _, row in sparse.iterrows():
                lines.append(
                    f"- Add hit/no-hit or fallback policy for borough `{row['borough']}` / segment `{row['property_segment']}` "
                    f"because observed row count is {int(row['n'])}."
                )
            lines.append("")
    lines.extend(
        [
            "## Architecture Hypotheses",
            "",
            "- Compare comps-only estimate, global XGBoost, segmented router, and residual-over-comps candidate on identical rows.",
            "- Add confidence/abstention using comp count, comp recency, comp dispersion, feature missingness, and slice residuals.",
            "- Add PLUTO only after current error and comp coverage artifacts show which property facts are missing.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_eda_report(
    *,
    profile: dict[str, Any],
    segment_region: pd.DataFrame,
    trends: pd.DataFrame,
    interactions: pd.DataFrame,
    error_slices: pd.DataFrame | None,
    hypotheses: str,
    predictions_csv: Path | None,
) -> str:
    top_segments = segment_region.head(8)
    strongest = interactions.head(10)
    report = [
        "# S.P.E.C. NYC Senior DS EDA Report",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Workflow Answer",
        "",
        "The frontend owns governed experiment lifecycle actions: hypothesis logging, review, queueing, worker start, governance, and package inspection. VS Code/CLI owns heavier EDA and feature research. This EDA job writes immutable artifacts under `reports/eda/`, so the analysis is still captured by the platform rather than living only in a notebook.",
        "",
        "## Dataset Profile",
        "",
        f"- Rows: {profile.get('row_count', 0):,}",
        f"- Sale window: {profile.get('sale_date_min')} to {profile.get('sale_date_max')}",
        f"- Median sale price: {_money(profile.get('sale_price', {}).get('median'))}",
        f"- Median PPSF: {_money(profile.get('price_per_sqft', {}).get('median'))}",
        f"- Unique properties: {profile.get('cardinality', {}).get('property_id', 'unknown')}",
        "",
        "## Segment And Region Structure",
        "",
        _markdown_table(
            top_segments,
            ["borough", "property_segment", "n", "median_sale_price", "median_ppsf", "median_sqft", "median_building_age"],
            max_rows=8,
        ),
        "",
        "## Real-Estate Specific Modeling Issues",
        "",
        "- Location effects are not globally stationary. Distance-to-center, density, and unit-count signals can change sign by borough and property type.",
        "- Property type should be treated as a model boundary, not only a feature. Single-family, small multifamily, walkup, elevator, condo/coop-like records have different comp logic.",
        "- Public records miss condition, renovation, views, floor level, layout, concessions, and listing demand. The model must expose confidence and abstention, not force a value.",
        "- Comparable-sales evidence should be both a model feature layer and a reviewer-facing explanation layer.",
        "",
        "## Feature Interaction Signals",
        "",
        _markdown_table(
            strongest,
            ["scope", "borough", "property_segment", "feature", "n", "spearman_corr_log_ppsf", "direction"],
            max_rows=10,
        ),
        "",
        "## Model Underperformance Signals",
        "",
    ]
    if error_slices is not None and not error_slices.empty:
        report.extend(
            [
                f"Prediction artifact analyzed: `{predictions_csv}`.",
                "",
                _markdown_table(
                    error_slices.head(10),
                    ["slice_type", "slice_name", "n", "mdape", "ppe10", "median_signed_pct_error", "overvaluation_rate"],
                    max_rows=10,
                ),
                "",
            ]
        )
    else:
        report.extend(
            [
                "No prediction artifact was supplied, so this run focuses on data readiness and modeling hypotheses.",
                "",
            ]
        )
    report.extend(
        [
            "## Architecture Recommendation",
            "",
            "Use a layered AVM architecture rather than one monolithic model:",
            "",
            "1. Comps-only transparent baseline.",
            "2. Global gradient-boosted tree model with point-in-time features.",
            "3. Segmented router by property type and, when data supports it, geography.",
            "4. Residual-over-comps model that predicts the correction from transparent market evidence.",
            "5. Confidence, hit/no-hit, and abstention layer from comp coverage, dispersion, feature completeness, and slice residuals.",
            "",
            "Deep learning should stay experimental until richer modalities exist, such as listing text, images, floor plans, or neighborhood embeddings.",
            "",
            "## Hypothesis Backlog",
            "",
            hypotheses.replace("# AVM EDA Hypothesis Backlog\n\n", ""),
            "## External Standards And Industry Context",
            "",
            "- IAAO describes AVM standards as principles and best practices for developing and using AVMs for real property valuation: https://researchexchange.iaao.org/jptaa/vol15/iss2/5/",
            "- Zillow says its Zestimate uses home facts, location, market trends, comparable homes, prior sales, and public/off-market records: https://zillow.zendesk.com/hc/en-us/articles/4402325964563-How-is-the-Zestimate-calculated",
            "- Fannie Mae emphasizes standardized, objective property data collection for AVM, appraisal, market analysis, and compliance use cases: https://www.fanniemae.com/research-and-insights/perspectives/advancing-collateral-valuation",
            "",
        ]
    )
    if not trends.empty:
        latest = trends.sort_values("period").tail(8)
        report.extend(
            [
                "## Latest Quarterly Market Trend Sample",
                "",
                _markdown_table(latest, ["period", "borough", "property_segment", "n", "median_ppsf", "ppsf_qoq_change"], max_rows=8),
                "",
            ]
        )
    return "\n".join(report)


def _available_usecols(path: Path, desired: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    lowered = {str(col).strip().lower(): col for col in header}
    return [lowered[col] for col in desired if col in lowered]


def _segment_columns() -> list[str]:
    return [
        "borough",
        "property_segment",
        "n",
        "row_share",
        "median_sale_price",
        "median_ppsf",
        "p25_ppsf",
        "p75_ppsf",
        "ppsf_iqr",
        "median_sqft",
        "median_building_age",
        "median_distance_to_center_km",
        "sqft_imputed_rate",
        "year_built_imputed_rate",
    ]


def _error_columns() -> list[str]:
    return [
        "slice_type",
        "slice_name",
        "n",
        "mdape",
        "ppe10",
        "ppe20",
        "median_signed_pct_error",
        "overvaluation_rate",
        "undervaluation_rate",
        "median_valuation_ratio",
        "median_comp_count",
        "median_comp_dispersion",
    ]


def _error_row(slice_type: str, slice_name: str, group: pd.DataFrame) -> dict[str, Any]:
    abs_error = pd.to_numeric(group["abs_pct_error"], errors="coerce").dropna()
    signed = pd.to_numeric(group["signed_pct_error"], errors="coerce").dropna()
    ratio = pd.to_numeric(group["valuation_ratio"], errors="coerce").dropna()
    comp_count = pd.to_numeric(group.get("comp_count"), errors="coerce")
    comp_dispersion = pd.to_numeric(group.get("comp_price_dispersion"), errors="coerce")
    return {
        "slice_type": slice_type,
        "slice_name": slice_name,
        "n": int(len(group)),
        "mdape": _median(abs_error),
        "ppe10": float((abs_error <= 0.10).mean()) if len(abs_error) else np.nan,
        "ppe20": float((abs_error <= 0.20).mean()) if len(abs_error) else np.nan,
        "median_signed_pct_error": _median(signed),
        "overvaluation_rate": float((signed > 0.10).mean()) if len(signed) else np.nan,
        "undervaluation_rate": float((signed < -0.10).mean()) if len(signed) else np.nan,
        "median_valuation_ratio": _median(ratio),
        "median_comp_count": _median(comp_count),
        "median_comp_dispersion": _median(comp_dispersion),
    }


def _feature_direction_flips(interactions: pd.DataFrame) -> list[tuple[str, str]]:
    flips = []
    scoped = interactions[interactions["scope"].isin(["borough_segment", "borough", "segment"])]
    for feature, group in scoped.groupby("feature"):
        directions = set(group["direction"].dropna().astype(str))
        if "positive" in directions and "negative" in directions:
            flips.append((str(feature), ", ".join(sorted(directions))))
    return flips


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3:
        return float("nan")
    xr = frame["x"].rank(method="average")
    yr = frame["y"].rank(method="average")
    if xr.nunique() < 2 or yr.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, float | None]:
    if column not in frame.columns:
        return {"missing": None}
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"missing": 1.0}
    return {
        "missing": float(pd.to_numeric(frame[column], errors="coerce").isna().mean()),
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def _date_bound(frame: pd.DataFrame, column: str, bound: str) -> str | None:
    if column not in frame.columns:
        return None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return str((values.min() if bound == "min" else values.max()).date())


def _median(values: Any) -> float:
    if values is None:
        return float("nan")
    if isinstance(values, (pd.Series, pd.Index, list, tuple, np.ndarray)):
        series = pd.to_numeric(values, errors="coerce")
    else:
        series = pd.to_numeric(pd.Series([values]), errors="coerce")
    series = pd.Series(series).dropna()
    return float(series.median()) if len(series) else float("nan")


def _quantile(values: Any, q: float) -> float:
    if values is None:
        return float("nan")
    series = pd.to_numeric(values, errors="coerce").dropna()
    return float(series.quantile(q)) if len(series) else float("nan")


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int) -> str:
    if frame.empty:
        return "_No rows available._"
    view = frame[[col for col in columns if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *body])


def _money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(number):
        return "unknown"
    return f"${number:,.0f}"


def _run_tag(tag: str | None) -> str:
    if tag:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in tag).strip("_") or "eda"
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate governed real-estate EDA artifacts.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = run_eda(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        tag=args.tag,
        predictions_csv=args.predictions_csv,
        limit=args.limit,
    )
    print(json.dumps({
        "report": str(result.report_path),
        "manifest": str(result.manifest_path),
        "hypothesis_backlog": str(result.hypothesis_backlog_path),
    }, indent=2))


if __name__ == "__main__":
    _cli()
