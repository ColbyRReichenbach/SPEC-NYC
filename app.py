"""S.P.E.C. NYC Streamlit dashboard (W3 functionalization)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from src.database import engine
from src.inference import (
    get_feature_columns,
    predict_dataframe,
    predict_single_row,
    select_pipeline_for_row,
)


st.set_page_config(
    page_title="S.P.E.C. NYC",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


BOROUGH_NAMES = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
}

SEGMENT_MAPPING = {
    "01": "SINGLE_FAMILY",
    "02": "SINGLE_FAMILY",
    "03": "SINGLE_FAMILY",
    "07": "WALKUP",
    "08": "ELEVATOR",
    "09": "WALKUP",
    "10": "ELEVATOR",
    "12": "WALKUP",
    "13": "ELEVATOR",
    "14": "SMALL_MULTI",
    "15": "SMALL_MULTI",
    "16": "SMALL_MULTI",
    "17": "SMALL_MULTI",
}


def _is_smoke_artifact(path: Path) -> bool:
    name = path.name.lower()
    return "smoke" in name or "dryrun" in name


def _latest_file(pattern: str, *, prefer_production: bool = True) -> Optional[Path]:
    files = list(Path(".").glob(pattern))
    if not files:
        return None

    if not prefer_production:
        return max(files, key=lambda p: p.stat().st_mtime)

    production_candidates = [p for p in files if not _is_smoke_artifact(p)]
    if production_candidates:
        return max(production_candidates, key=lambda p: p.stat().st_mtime)
    return max(files, key=lambda p: p.stat().st_mtime)


def _fmt_currency(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"${value:,.0f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.1f}%"


def _age_days_from_date(date_value: Any) -> Optional[int]:
    if date_value is None:
        return None
    date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date):
        return None
    return int((datetime.utcnow().date() - date.date()).days)


def _status_label(ok: bool, detail: str) -> str:
    return ("OK: " if ok else "WARN: ") + detail


@st.cache_data(show_spinner=False)
def load_metrics() -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    path = _latest_file("models/metrics_*.json")
    if path is None:
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except Exception:
        return None, path


@st.cache_data(show_spinner=False)
def load_segment_scorecard() -> Tuple[pd.DataFrame, Optional[Path]]:
    path = _latest_file("reports/model/segment_scorecard_*.csv")
    if path is None:
        return pd.DataFrame(), None
    try:
        return pd.read_csv(path), path
    except Exception:
        return pd.DataFrame(), path


@st.cache_data(show_spinner=False)
def load_eval_predictions() -> Tuple[pd.DataFrame, Optional[Path]]:
    path = _latest_file("reports/model/evaluation_predictions_*.csv")
    if path is None:
        return pd.DataFrame(), None
    try:
        df = pd.read_csv(path)
        if "sale_date" in df.columns:
            df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
        return df, path
    except Exception:
        return pd.DataFrame(), path


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[str]]:
    path = _latest_file("models/model_*.joblib")
    if path is None:
        return None, None, "No model artifact found."
    try:
        return joblib.load(path), path, None
    except Exception as exc:
        return None, path, f"Failed to load model artifact: {exc}"


def _normalize_sales_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    rename_map = {
        "building_class_at_time_of": "building_class",
        "tax_class_at_time_of_sale": "tax_class",
        "census_tract_2020": "census_tract",
    }
    frame = frame.rename(columns=rename_map)
    if "sale_date" in frame.columns:
        frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    if "borough" in frame.columns:
        frame["borough_name"] = frame["borough"].map(BOROUGH_NAMES).fillna(frame["borough"].astype(str))
    else:
        frame["borough_name"] = "Unknown"
    if "building_class_category" in frame.columns and "property_segment" not in frame.columns:
        prefix = frame["building_class_category"].astype(str).str[:2]
        frame["property_segment"] = prefix.map(SEGMENT_MAPPING).fillna("OTHER")
    if "property_segment" not in frame.columns:
        frame["property_segment"] = "OTHER"
    return frame


@st.cache_data(show_spinner=False)
def load_sales_data(limit: int = 10000) -> Tuple[pd.DataFrame, str, Optional[str]]:
    query = text(
        """
        SELECT
            id,
            bbl,
            borough,
            block,
            lot,
            address,
            apartment_number,
            neighborhood,
            latitude,
            longitude,
            building_class_category,
            building_class,
            residential_units,
            total_units,
            gross_square_feet,
            year_built,
            sale_price,
            sale_date,
            h3_index,
            distance_to_center_km,
            property_id,
            property_segment,
            price_tier,
            building_age,
            is_latest_sale
        FROM sales
        ORDER BY sale_date DESC
        LIMIT :limit
        """
    )
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
        return _normalize_sales_columns(df), "postgres", None
    except Exception as db_exc:
        raw_path = Path("data/raw/annualized_sales_2019_2025.csv")
        if raw_path.exists():
            try:
                df = pd.read_csv(raw_path, low_memory=False, nrows=limit)
                return _normalize_sales_columns(df), "raw_csv", f"Postgres unavailable: {db_exc}"
            except Exception as csv_exc:
                return pd.DataFrame(), "none", f"DB error: {db_exc}; CSV fallback error: {csv_exc}"
        return pd.DataFrame(), "none", f"Data source unavailable: {db_exc}"


def _with_h3_price_lag(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "h3_price_lag" in out.columns and out["h3_price_lag"].notna().any():
        return out
    if "h3_index" in out.columns and "sale_price" in out.columns:
        med = out.groupby("h3_index")["sale_price"].median()
        global_med = float(pd.to_numeric(out["sale_price"], errors="coerce").median())
        out["h3_price_lag"] = out["h3_index"].map(med).fillna(global_med)
    else:
        out["h3_price_lag"] = np.nan
    return out


def _build_valuation_frame(
    sales_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_artifact: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, str]:
    if model_artifact is None:
        return pd.DataFrame(), "none"

    feature_cols = get_feature_columns(model_artifact)
    source = "evaluation_predictions"
    candidate = eval_df.copy() if not eval_df.empty else sales_df.copy()
    if candidate.empty:
        return pd.DataFrame(), "none"

    candidate = _with_h3_price_lag(_normalize_sales_columns(candidate))
    missing = [c for c in feature_cols if c not in candidate.columns]
    if missing:
        return pd.DataFrame(), "missing_features"

    if "sale_date" in candidate.columns:
        candidate["sale_date"] = pd.to_datetime(candidate["sale_date"], errors="coerce")
    candidate = candidate.sort_values("sale_date", ascending=False) if "sale_date" in candidate.columns else candidate
    candidate = candidate.reset_index(drop=True)
    candidate["row_id"] = candidate.index.astype(int)

    if "property_id" in candidate.columns:
        label = candidate["property_id"].astype(str)
    elif {"address", "apartment_number"}.issubset(candidate.columns):
        label = candidate["address"].fillna("Unknown").astype(str) + " " + candidate["apartment_number"].fillna("").astype(str)
    elif "bbl" in candidate.columns:
        label = "BBL " + candidate["bbl"].astype(str)
    else:
        label = "row_" + candidate.index.astype(str)

    if "sale_date" in candidate.columns:
        candidate["display_label"] = label + " | " + candidate["sale_date"].dt.strftime("%Y-%m-%d").fillna("unknown_date")
    else:
        candidate["display_label"] = label

    if eval_df.empty:
        source = "sales"
    return candidate, source


def predict_row(
    model_artifact: Dict[str, Any],
    row: pd.Series,
) -> Tuple[Optional[float], Optional[str]]:
    try:
        pred, _route = predict_single_row(model_artifact, row)
        return pred, None
    except Exception as exc:
        return None, str(exc)


def explain_row(
    model_artifact: Dict[str, Any],
    row: pd.Series,
) -> Tuple[pd.DataFrame, Optional[float], Optional[str]]:
    try:
        import xgboost as xgb
    except Exception as exc:
        return pd.DataFrame(), None, f"xgboost import failed: {exc}"

    try:
        feature_cols = get_feature_columns(model_artifact)
        pipeline, _ = select_pipeline_for_row(model_artifact, row)
        pre = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        x = pd.DataFrame([row[feature_cols].to_dict()])
        transformed = pre.transform(x)
        dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        feature_names = pre.get_feature_names_out()
        dmatrix = xgb.DMatrix(dense, feature_names=list(feature_names))
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)[0]
        contrib_df = pd.DataFrame(
            {
                "feature": feature_names,
                "contribution": contribs[:-1],
            }
        )
        contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
        contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)
        base_value = float(contribs[-1])
        return contrib_df, base_value, None
    except Exception as exc:
        return pd.DataFrame(), None, str(exc)


def render_waterfall(contrib_df: pd.DataFrame, base_value: float) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        import shap

        exp = shap.Explanation(
            values=contrib_df["contribution"].values,
            base_values=base_value,
            data=np.zeros(len(contrib_df)),
            feature_names=contrib_df["feature"].tolist(),
        )
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(exp, max_display=12, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return None
    except Exception as exc:
        return str(exc)


def add_valuation_status(
    frame: pd.DataFrame,
    model_artifact: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Optional[str]]:
    if frame.empty or model_artifact is None:
        return frame, None
    out = _with_h3_price_lag(_normalize_sales_columns(frame))
    feature_cols = get_feature_columns(model_artifact)
    missing = [c for c in feature_cols if c not in out.columns]
    if missing:
        return out, None
    try:
        preds, routes = predict_dataframe(model_artifact, out)
        out["predicted_price"] = preds
        out["model_route"] = routes.values
        if "sale_price" in out.columns:
            diff = (pd.to_numeric(out["sale_price"], errors="coerce") - out["predicted_price"]) / out["predicted_price"]
            out["valuation_status"] = np.select(
                [diff >= 0.10, diff <= -0.10],
                ["over_valued", "under_valued"],
                default="near_fair_value",
            )
            return out, "valuation_status"
        return out, "predicted_price"
    except Exception:
        return out, None


def render_pipeline_status(
    sales_df: pd.DataFrame,
    metrics_path: Optional[Path],
    model_path: Optional[Path],
    scorecard_path: Optional[Path],
) -> None:
    latest_etl = _latest_file("reports/data/etl_run_*.md")
    latest_shap_summary = _latest_file("reports/model/shap_summary_*.png")

    latest_sale_date = None
    if "sale_date" in sales_df.columns and not sales_df.empty:
        latest_sale_date = pd.to_datetime(sales_df["sale_date"], errors="coerce").max()
    freshness_days = _age_days_from_date(latest_sale_date)

    status_rows = []
    status_rows.append(
        {
            "component": "Sales data",
            "status": _status_label(
                sales_df.shape[0] > 0 and freshness_days is not None and freshness_days <= 730,
                f"rows={sales_df.shape[0]}, latest_sale_date={latest_sale_date.date() if pd.notna(latest_sale_date) else 'N/A'}",
            ),
        }
    )
    status_rows.append(
        {
            "component": "ETL report",
            "status": _status_label(latest_etl is not None, str(latest_etl) if latest_etl else "missing"),
        }
    )
    status_rows.append(
        {
            "component": "Model artifact",
            "status": _status_label(model_path is not None, str(model_path) if model_path else "missing"),
        }
    )
    status_rows.append(
        {
            "component": "Metrics and scorecard",
            "status": _status_label(metrics_path is not None and scorecard_path is not None, "available" if metrics_path and scorecard_path else "missing"),
        }
    )
    status_rows.append(
        {
            "component": "SHAP assets",
            "status": _status_label(latest_shap_summary is not None, str(latest_shap_summary) if latest_shap_summary else "missing"),
        }
    )

    st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)


def main() -> None:
    st.title("S.P.E.C. NYC Valuation Engine")
    st.caption("Spatial | Predictive | Explainable | Conversational")

    if st.sidebar.button("Reload Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    metrics, metrics_path = load_metrics()
    scorecard_df, scorecard_path = load_segment_scorecard()
    eval_df, eval_path = load_eval_predictions()
    sales_df, sales_source, sales_error = load_sales_data(limit=15000)
    model_artifact, model_path, model_error = load_model_artifact()

    if sales_error:
        st.warning(sales_error)
    if model_error:
        st.warning(model_error)

    overall = (metrics or {}).get("overall", {})
    metadata = (metrics or {}).get("metadata", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PPE10", _fmt_pct(overall.get("ppe10")))
    c2.metric("MdAPE", _fmt_pct(overall.get("mdape")))
    c3.metric("R2", f"{overall.get('r2', float('nan')):.3f}" if "r2" in overall else "N/A")
    c4.metric("Eval Rows", f"{int(overall['n']):,}" if "n" in overall else "N/A")

    st.markdown("### Data Recency and Pipeline Status")
    st.write(
        f"Data source: `{sales_source}` | Metrics: `{metrics_path or 'missing'}` | "
        f"Eval predictions: `{eval_path or 'missing'}`"
    )
    render_pipeline_status(sales_df, metrics_path, model_path, scorecard_path)

    st.markdown("### Segment Health")
    if scorecard_df.empty:
        st.info("Segment scorecard not found. Run `python3 -m src.model` or `python3 -m src.evaluate`.")
    else:
        segments = scorecard_df[scorecard_df["group_type"] == "segment"].copy() if "group_type" in scorecard_df.columns else pd.DataFrame()
        if segments.empty:
            st.info("No segment rows found in scorecard.")
        else:
            fig_seg = px.bar(
                segments.sort_values("ppe10", ascending=False),
                x="group_name",
                y="ppe10",
                color="group_name",
                labels={"group_name": "Segment", "ppe10": "PPE10"},
                title="PPE10 by Segment",
            )
            fig_seg.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig_seg, use_container_width=True)
            st.dataframe(segments, hide_index=True, use_container_width=True)

    st.markdown("### Sales Explorer")
    map_df = sales_df.copy()
    if not map_df.empty and "is_latest_sale" in map_df.columns:
        default_latest_only = st.sidebar.checkbox("Show latest sales only", value=True)
        if default_latest_only:
            map_df = map_df[map_df["is_latest_sale"] == True]  # noqa: E712

    if not map_df.empty:
        borough_options = sorted(map_df["borough_name"].dropna().astype(str).unique().tolist())
        selected_boroughs = st.sidebar.multiselect("Borough Filter", borough_options, default=borough_options[:2] if borough_options else [])
        if selected_boroughs:
            map_df = map_df[map_df["borough_name"].isin(selected_boroughs)]

        if "property_segment" in map_df.columns:
            segment_options = sorted(map_df["property_segment"].dropna().astype(str).unique().tolist())
            selected_segments = st.sidebar.multiselect("Segment Filter", segment_options, default=segment_options)
            if selected_segments:
                map_df = map_df[map_df["property_segment"].isin(selected_segments)]

    map_ready = not map_df.empty and {"latitude", "longitude"}.issubset(map_df.columns)
    if map_ready:
        coords = map_df.dropna(subset=["latitude", "longitude"]).copy()
        coords, map_color_col = add_valuation_status(coords, model_artifact)
        if coords.empty:
            st.info("No coordinate rows available after filters.")
        else:
            fig_map = px.scatter_mapbox(
                coords.head(4000),
                lat="latitude",
                lon="longitude",
                color=map_color_col if map_color_col else ("property_segment" if "property_segment" in coords.columns else None),
                size="sale_price" if "sale_price" in coords.columns else None,
                size_max=12,
                hover_name="address" if "address" in coords.columns else None,
                hover_data=["sale_price", "sale_date", "borough_name", "neighborhood"] if "sale_price" in coords.columns else None,
                zoom=9,
                height=450,
            )
            fig_map.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)
            table_cols = [c for c in ["sale_date", "sale_price", "borough_name", "property_segment", "neighborhood", "address"] if c in coords.columns]
            st.dataframe(coords[table_cols].head(200), use_container_width=True, hide_index=True)
    else:
        st.info("Map unavailable. Missing sales coordinates or source data.")

    st.markdown("### Per-Property Valuation and Explainability")
    valuation_df, valuation_source = _build_valuation_frame(sales_df, eval_df, model_artifact)
    if valuation_df.empty or model_artifact is None:
        st.info("Valuation panel unavailable. Ensure model artifact and feature-complete data are present.")
        return

    st.write(f"Valuation source: `{valuation_source}` | model: `{model_path}`")
    options = valuation_df["display_label"].head(1000).tolist()
    selected_label = st.selectbox("Select property", options=options)
    selected_row = valuation_df.loc[valuation_df["display_label"] == selected_label].iloc[0]

    pred_price, pred_error = predict_row(model_artifact, selected_row)
    if pred_error:
        st.error(f"Prediction failed: {pred_error}")
        return

    actual_price = pd.to_numeric(selected_row.get("sale_price"), errors="coerce") if "sale_price" in selected_row.index else np.nan
    mdape_overall = float(overall.get("mdape", 0.10)) if overall else 0.10
    mdape_used = mdape_overall
    segment_name = str(selected_row.get("property_segment", ""))
    per_segment = (metrics or {}).get("per_segment", {})
    if segment_name in per_segment:
        mdape_used = float(per_segment[segment_name].get("mdape", mdape_overall))
    lower = pred_price * (1 - mdape_used)
    upper = pred_price * (1 + mdape_used)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Predicted Value", _fmt_currency(pred_price))
    v2.metric("Range (+/- MdAPE)", f"{_fmt_currency(lower)} to {_fmt_currency(upper)}")
    v3.metric("Actual Sale Price", _fmt_currency(actual_price) if not np.isnan(actual_price) else "N/A")
    if not np.isnan(actual_price):
        v4.metric("Prediction Error", _fmt_currency(pred_price - actual_price))
    else:
        v4.metric("Prediction Error", "N/A")

    st.write(
        f"Uncertainty note: using MdAPE={mdape_used * 100:.2f}% "
        f"({'segment-specific' if segment_name in per_segment else 'overall baseline'})."
    )

    if not sales_df.empty:
        history = pd.DataFrame()
        if "property_id" in selected_row.index and pd.notna(selected_row.get("property_id")) and "property_id" in sales_df.columns:
            history = sales_df[sales_df["property_id"] == selected_row["property_id"]].copy()
        elif "bbl" in selected_row.index and pd.notna(selected_row.get("bbl")) and "bbl" in sales_df.columns:
            history = sales_df[sales_df["bbl"] == selected_row["bbl"]].copy()
        if not history.empty:
            history = history.sort_values("sale_date", ascending=False) if "sale_date" in history.columns else history
            cols = [c for c in ["sale_date", "sale_price", "price_change_pct", "sale_sequence", "is_latest_sale", "address"] if c in history.columns]
            st.markdown("#### Full Price History")
            st.dataframe(history[cols], hide_index=True, use_container_width=True)

    contrib_df, base_value, explain_error = explain_row(model_artifact, selected_row)
    if explain_error:
        st.warning(f"Explainability unavailable for this row: {explain_error}")
        return
    top_contrib = contrib_df.head(12).copy()
    top_contrib["direction"] = np.where(top_contrib["contribution"] >= 0, "up", "down")
    fig_contrib = px.bar(
        top_contrib.sort_values("contribution"),
        x="contribution",
        y="feature",
        color="direction",
        orientation="h",
        title="Top SHAP-like feature contributions (local)",
    )
    st.plotly_chart(fig_contrib, use_container_width=True)
    st.dataframe(top_contrib[["feature", "contribution"]], hide_index=True, use_container_width=True)

    if base_value is not None:
        st.markdown("#### SHAP Waterfall")
        waterfall_error = render_waterfall(contrib_df, base_value)
        if waterfall_error:
            st.info(f"Waterfall rendering unavailable: {waterfall_error}")


if __name__ == "__main__":
    main()
