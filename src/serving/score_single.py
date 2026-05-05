"""Model-backed single-property scoring for the dashboard API.

This module is intentionally a small process boundary. The Next.js API passes a
validated request and package path; Python loads the actual model package,
builds inference features, and returns only JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.comps import compute_asof_comps_for_frame
from src.inference import predict_dataframe, _with_temporal_regime_features  # noqa: PLC2701
from src.pre_comps import add_sale_validity_labels, _compute_asof_features  # noqa: PLC2701


BOROUGH_TO_CODE = {
    "MANHATTAN": "1",
    "NEW_YORK": "1",
    "BRONX": "2",
    "BROOKLYN": "3",
    "KINGS": "3",
    "QUEENS": "4",
    "STATEN_ISLAND": "5",
    "STATEN ISLAND": "5",
    "RICHMOND": "5",
}


def score_single(repo_root: Path, package_path: str, request: dict[str, Any]) -> dict[str, Any]:
    package_dir = _safe_package_dir(repo_root, package_path)
    model_artifact = joblib.load(package_dir / "model.joblib")
    metrics = _read_json(package_dir / "metrics.json")
    data_manifest = _read_json(package_dir / "data_manifest.json")
    feature_contract = _read_json(package_dir / "feature_contract.json")
    explainability = _read_json(package_dir / "explainability_manifest.json")

    row = _build_base_row(request)
    feature_frame, feature_evidence = _build_feature_frame(repo_root, data_manifest, row)
    feature_frame = _with_temporal_regime_features(feature_frame)
    preds, routes = predict_dataframe(model_artifact, feature_frame)
    predicted_price = float(preds[0])
    route = str(routes.iloc[0])
    drivers = _local_contributions(model_artifact, feature_frame)
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    per_segment = metrics.get("per_segment", {}) if isinstance(metrics, dict) else {}
    segment = str(request["property"]["property_segment"])
    segment_metrics = per_segment.get(segment, {}) if isinstance(per_segment, dict) else {}
    mdape = _number(segment_metrics.get("mdape"), _number(overall.get("mdape"), 0.35))
    ppe10 = _number(segment_metrics.get("ppe10"), _number(overall.get("ppe10"), 0.0))
    interval_margin = max(0.05, min(0.75, mdape))

    expected_features = _feature_names(feature_contract, model_artifact)
    missing_features = [name for name in expected_features if name not in feature_frame.columns or pd.isna(feature_frame.iloc[0].get(name))]
    completeness = 1.0 - (len(missing_features) / max(len(expected_features), 1))
    confidence_score = max(0.0, min(1.0, 0.45 * ppe10 + 0.35 * (1.0 - mdape) + 0.20 * completeness))

    return {
        "predicted_price": round(predicted_price),
        "prediction_interval": {
            "low": round(predicted_price * (1.0 - interval_margin)),
            "high": round(predicted_price * (1.0 + interval_margin)),
            "method": "artifact_metric_mdape_band_v1",
        },
        "confidence": {
            "score": round(confidence_score, 4),
            "band": "high" if confidence_score >= 0.75 else "medium" if confidence_score >= 0.45 else "low",
            "factors": {
                "segment_calibration": round(max(0.0, min(1.0, ppe10)), 4),
                "support_coverage": round(_support_coverage(feature_frame.iloc[0]), 4),
                "input_completeness": round(completeness, 4),
            },
            "caveats": _caveats(missing_features, mdape, feature_evidence),
        },
        "explanation": {
            "status": "ready" if drivers["positive"] or drivers["negative"] else "degraded",
            "explainer_type": "xgboost_pred_contribs_grouped_to_contract_features",
            "local_accuracy": None,
            "drivers_positive": drivers["positive"],
            "drivers_negative": drivers["negative"],
        },
        "model": {
            "run_id": str(metrics.get("metadata", {}).get("artifact_tag") or package_dir.name),
            "model_version": str(metrics.get("metadata", {}).get("model_version") or "unknown"),
            "route": route,
            "model_package_id": str(metrics.get("metadata", {}).get("model_package_id") or package_dir.name),
        },
        "evidence": {
            "model_package_path": _rel(repo_root, package_dir),
            "metrics_path": _rel(repo_root, package_dir / "metrics.json"),
            "feature_contract_path": _rel(repo_root, package_dir / "feature_contract.json"),
            "data_manifest_path": _rel(repo_root, package_dir / "data_manifest.json"),
            "model_card_path": _rel(repo_root, package_dir / "model_card.md"),
            "feature_importance_artifact": str(explainability.get("feature_importance_artifact") or "not_generated"),
            "training_source_path": feature_evidence["training_source_path"],
            "feature_vector_sha256": _sha256_json(_jsonable(feature_frame.iloc[0].to_dict())),
            "missing_features": missing_features,
            "feature_generation": feature_evidence,
        },
    }


def global_importance(repo_root: Path, package_path: str, segment: str, window: str) -> dict[str, Any]:
    package_dir = _safe_package_dir(repo_root, package_path)
    model_artifact = joblib.load(package_dir / "model.joblib")
    metrics = _read_json(package_dir / "metrics.json")
    feature_contract = _read_json(package_dir / "feature_contract.json")
    pipeline = model_artifact.get("fallback_pipeline") or model_artifact.get("pipeline")
    if pipeline is None:
        raise RuntimeError("Model artifact has no pipeline for importance extraction.")
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    importances = getattr(model, "feature_importances_", None)
    preprocessor = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None
    if importances is None or preprocessor is None:
        raise RuntimeError("Model artifact does not expose feature importances.")

    transformed_names = _transformed_feature_names(preprocessor)
    grouped: dict[str, float] = {}
    for name, value in zip(transformed_names, importances):
        grouped[_contract_feature_name(name, feature_contract)] = grouped.get(_contract_feature_name(name, feature_contract), 0.0) + float(value)
    rows = [
        {
            "feature_name": feature,
            "mean_abs_shap": round(value, 8),
            "direction_hint": "mixed",
        }
        for feature, value in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
        if value > 0
    ][:20]
    return {
        "segment": segment.upper(),
        "window": window,
        "features": rows,
        "generated_from": [
            _rel(repo_root, package_dir / "model.joblib"),
            _rel(repo_root, package_dir / "feature_contract.json"),
            _rel(repo_root, package_dir / "metrics.json"),
        ],
        "model_package_id": str(metrics.get("metadata", {}).get("model_package_id") or package_dir.name),
    }


def _safe_package_dir(repo_root: Path, package_path: str) -> Path:
    package_dir = (repo_root / package_path).resolve()
    packages_root = (repo_root / "models/packages").resolve()
    if not str(package_dir).startswith(str(packages_root)):
        raise ValueError("Package path is outside models/packages.")
    if not (package_dir / "model.joblib").exists():
        raise FileNotFoundError(f"Model package is missing model.joblib: {package_path}")
    return package_dir


def _build_base_row(request: dict[str, Any]) -> pd.DataFrame:
    prop = request["property"]
    sale_date = pd.to_datetime(prop["sale_date"], errors="coerce")
    if pd.isna(sale_date):
        sale_date = pd.Timestamp(datetime.utcnow().date())
    borough = _normalize_borough(prop["borough"])
    year_built = _number(prop.get("year_built"), np.nan)
    building_age = float(sale_date.year - year_built) if np.isfinite(year_built) else np.nan
    return pd.DataFrame(
        [
            {
                "property_id": request.get("context", {}).get("property_id") or _property_id(prop),
                "sale_date": sale_date.strftime("%Y-%m-%d"),
                "h3_index": prop.get("h3_index"),
                "gross_square_feet": _number(prop.get("gross_square_feet"), np.nan),
                "year_built": year_built,
                "building_age": building_age,
                "residential_units": _number(prop.get("residential_units"), np.nan),
                "total_units": _number(prop.get("total_units"), np.nan),
                "distance_to_center_km": _number(prop.get("distance_to_center_km"), np.nan),
                "borough": borough,
                "building_class": str(prop.get("building_class") or "").strip().upper(),
                "property_segment": str(prop.get("property_segment") or "").strip().upper(),
                "neighborhood": str(prop.get("neighborhood") or "UNKNOWN").strip().upper(),
                "latitude": _number(prop.get("latitude"), np.nan),
                "longitude": _number(prop.get("longitude"), np.nan),
            }
        ]
    )


def _build_feature_frame(repo_root: Path, data_manifest: dict[str, Any], row: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = _training_source_path(data_manifest)
    evidence: dict[str, Any] = {
        "training_source_path": source_path,
        "h3_features_status": "not_generated",
        "comps_features_status": "not_generated",
        "selected_comps": [],
    }
    frame = row.copy()
    if not source_path:
        return frame, evidence

    full_source = repo_root / source_path
    if not full_source.exists():
        evidence["training_source_status"] = "missing"
        return frame, evidence

    reference = pd.read_csv(full_source, low_memory=False)
    if "sale_validity_status" not in reference.columns:
        reference = add_sale_validity_labels(reference)

    try:
        h3_features = _compute_asof_features(frame, reference, min_h3_prior_count=3)
        for column in h3_features.columns:
            frame[column] = h3_features[column].values
        evidence["h3_features_status"] = "generated"
    except Exception as exc:
        evidence["h3_features_status"] = f"degraded: {exc}"

    try:
        comp_features, selected_comps, manifest = compute_asof_comps_for_frame(
            frame,
            reference,
            split_name="scoring",
            include_selected=True,
        )
        for column in comp_features.columns:
            frame[column] = comp_features[column].values
        evidence["comps_features_status"] = "generated"
        evidence["comps_manifest"] = manifest
        evidence["selected_comps"] = _jsonable(selected_comps.head(10).to_dict(orient="records"))
    except Exception as exc:
        evidence["comps_features_status"] = f"degraded: {exc}"

    return frame, evidence


def _training_source_path(data_manifest: dict[str, Any]) -> str | None:
    sources = data_manifest.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict) and source.get("uri"):
                return str(source["uri"])
    return None


def _local_contributions(model_artifact: dict[str, Any], feature_frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    pipeline = model_artifact.get("fallback_pipeline") or model_artifact.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "named_steps"):
        return {"positive": [], "negative": []}
    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None or not hasattr(model, "get_booster"):
        return {"positive": [], "negative": []}

    feature_columns = list(model_artifact.get("feature_columns") or [])
    transformed = preprocessor.transform(feature_frame[feature_columns])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    try:
        import xgboost as xgb

        contribs = model.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[0][:-1]
    except Exception:
        return {"positive": [], "negative": []}

    transformed_names = _transformed_feature_names(preprocessor)
    grouped: dict[str, float] = {}
    for name, value in zip(transformed_names, contribs):
        feature = _contract_feature_name(name, {"features": [{"name": item} for item in feature_columns]})
        grouped[feature] = grouped.get(feature, 0.0) + float(value)

    rows = [
        {
            "feature": feature,
            "impact": round(value),
            "display": feature.replace("_", " "),
        }
        for feature, value in grouped.items()
        if np.isfinite(value) and abs(value) >= 1.0
    ]
    positive = sorted((row for row in rows if row["impact"] > 0), key=lambda row: abs(row["impact"]), reverse=True)[:8]
    negative = sorted((row for row in rows if row["impact"] < 0), key=lambda row: abs(row["impact"]), reverse=True)[:8]
    return {"positive": positive, "negative": negative}


def _transformed_feature_names(preprocessor: Any) -> list[str]:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def _contract_feature_name(transformed_name: str, feature_contract: dict[str, Any]) -> str:
    clean = transformed_name.replace("num__", "").replace("cat__", "")
    feature_names = _feature_names(feature_contract, {})
    for feature in sorted(feature_names, key=len, reverse=True):
        if clean == feature or clean.startswith(f"{feature}_"):
            return feature
    return clean


def _feature_names(feature_contract: dict[str, Any], model_artifact: dict[str, Any]) -> list[str]:
    features = feature_contract.get("features")
    if isinstance(features, list):
        names = [str(item.get("name")) for item in features if isinstance(item, dict) and item.get("name")]
        if names:
            return names
    return [str(item) for item in model_artifact.get("feature_columns", [])]


def _support_coverage(row: pd.Series) -> float:
    comp_count = _number(row.get("comp_count"), 0.0)
    h3_count = _number(row.get("h3_prior_sale_count"), 0.0)
    comp_support = min(1.0, comp_count / 3.0)
    h3_support = min(1.0, h3_count / 3.0)
    return max(comp_support, h3_support)


def _caveats(missing_features: list[str], mdape: float, evidence: dict[str, Any]) -> list[str]:
    caveats = ["Estimate is model-backed and probabilistic, not an appraisal."]
    if missing_features:
        caveats.append(f"{len(missing_features)} contract features were missing and handled by the trained preprocessing pipeline.")
    if mdape >= 0.25:
        caveats.append("Validation error for this segment/package is elevated; use the interval and comparable evidence.")
    if str(evidence.get("comps_features_status", "")).startswith("degraded"):
        caveats.append("Comparable-sales features degraded during scoring; inspect feature-generation evidence.")
    return caveats


def _normalize_borough(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    return BOROUGH_TO_CODE.get(text, text)


def _property_id(prop: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(prop.get("address") or ""),
            str(prop.get("borough") or ""),
            str(prop.get("sale_date") or ""),
            str(prop.get("gross_square_feet") or ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _number(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_jsonable(entry) for entry in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rel(repo_root: Path, path_value: Path) -> str:
    return str(path_value.resolve().relative_to(repo_root.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--mode", choices=["score", "global-importance"], default="score")
    parser.add_argument("--segment", default="ALL")
    parser.add_argument("--window", default="180d")
    args = parser.parse_args()

    try:
        if args.mode == "score":
            request = json.loads(sys.stdin.read() or "{}")
            payload = score_single(args.repo_root.resolve(), args.package_path, request)
        else:
            payload = global_importance(args.repo_root.resolve(), args.package_path, args.segment, args.window)
        print(json.dumps(_jsonable(payload), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
