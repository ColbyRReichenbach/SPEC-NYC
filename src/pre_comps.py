"""Pre-comps DS controls for leakage-safe AVM training.

This module contains the feature and manifest logic that must be correct before
we build a comparable-sales engine. The functions here avoid target leakage by
only using historical rows strictly before the valuation/sale date.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DATE_COL = "sale_date"
TARGET_COL = "sale_price"

LOCAL_MARKET_FEATURES = [
    "h3_prior_sale_count",
    "h3_prior_median_price",
    "h3_prior_median_ppsf",
]

MODEL_CRITICAL_IMPUTED_FIELDS = {
    "gross_square_feet": "sqft_imputed",
    "year_built": "year_built_imputed",
}

PRE_COMPS_FEATURE_AVAILABILITY = {
    "gross_square_feet": {
        "source": "public_record_or_train_fit_imputation",
        "as_of_policy": "ETL-imputed values are reset to missing before model fitting; imputation is train-fit only.",
    },
    "year_built": {
        "source": "public_record_or_train_fit_imputation",
        "as_of_policy": "ETL-imputed values are reset to missing before model fitting; imputation is train-fit only.",
    },
    "building_age": {
        "source": "derived_from_year_built_and_valuation_date",
        "as_of_policy": "Set missing when year_built was ETL-imputed; train-fit imputation handles fallback.",
    },
    "h3_prior_sale_count": {
        "source": "historical_public_sales",
        "as_of_policy": "Uses same-H3 sales strictly before the row sale/valuation date.",
    },
    "h3_prior_median_price": {
        "source": "historical_public_sales",
        "as_of_policy": "Uses same-H3 sale prices strictly before the row sale/valuation date; never current/future target.",
    },
    "h3_prior_median_ppsf": {
        "source": "historical_public_sales",
        "as_of_policy": "Uses same-H3 historical PPSF strictly before the row sale/valuation date.",
    },
    "comp_count": {
        "source": "as_of_comparable_sales_engine",
        "as_of_policy": "Counts selected valid comparable sales strictly before the row sale/valuation date.",
    },
    "comp_median_price": {
        "source": "as_of_comparable_sales_engine",
        "as_of_policy": "Median price from selected valid comparable sales strictly before valuation.",
    },
    "comp_median_ppsf": {
        "source": "as_of_comparable_sales_engine",
        "as_of_policy": "Median PPSF from selected valid comparable sales strictly before valuation.",
    },
    "comp_weighted_estimate": {
        "source": "as_of_comparable_sales_engine",
        "as_of_policy": "Similarity-weighted estimate from selected valid comparable sales strictly before valuation.",
    },
    "comp_price_dispersion": {
        "source": "as_of_comparable_sales_engine",
        "as_of_policy": "Dispersion of selected comparable prices/PPSF computed from historical comps only.",
    },
    "comp_nearest_distance_km": {
        "source": "as_of_comparable_sales_engine",
        "as_of_policy": "Nearest selected historical comparable distance.",
    },
    "comp_median_recency_days": {
        "source": "as_of_comparable_sales_engine",
        "as_of_policy": "Median age in days of selected historical comparable sales.",
    },
    "comp_local_momentum": {
        "source": "as_of_comparable_sales_engine",
        "as_of_policy": "Recent-vs-older selected comparable PPSF trend computed from historical comps only.",
    },
}


@dataclass(frozen=True)
class AsOfFeatureResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    manifest: dict[str, Any]


def add_sale_validity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add auditable sale-validity labels for training and future comp eligibility."""
    out = df.copy()
    sale_price = pd.to_numeric(out.get(TARGET_COL), errors="coerce")
    sqft = pd.to_numeric(out.get("gross_square_feet"), errors="coerce")
    ppsf = sale_price / sqft.where(sqft > 0)

    reasons: list[list[str]] = [[] for _ in range(len(out))]
    exclude = np.zeros(len(out), dtype=bool)

    def add_reason(mask: pd.Series | np.ndarray, reason: str, *, exclude_training: bool = False) -> None:
        bool_mask = np.asarray(mask, dtype=bool)
        for idx in np.flatnonzero(bool_mask):
            reasons[idx].append(reason)
        if exclude_training:
            exclude[:] = exclude | bool_mask

    add_reason(sale_price.isna(), "missing_sale_price", exclude_training=True)
    add_reason(sale_price < 10_000, "below_min_market_sale_price", exclude_training=True)
    add_reason(sale_price > 100_000_000, "extreme_sale_price_exclude", exclude_training=True)
    add_reason((sale_price > 50_000_000) & (sale_price <= 100_000_000), "extreme_sale_price_review")

    add_reason(sqft.isna() | (sqft <= 0), "missing_or_nonpositive_sqft_review")
    add_reason((ppsf < 50) & ppsf.notna(), "extreme_low_ppsf_review")
    add_reason((ppsf > 5_000) & ppsf.notna(), "extreme_high_ppsf_review")

    if {"bbl", DATE_COL, TARGET_COL}.issubset(out.columns):
        duplicate_bbl_sale = out.duplicated(["bbl", DATE_COL, TARGET_COL], keep=False)
        if "property_id" in out.columns:
            duplicate_property_sale = out.duplicated(["property_id", DATE_COL, TARGET_COL], keep=False)
            unresolved_duplicate = duplicate_bbl_sale & ~duplicate_property_sale
        else:
            unresolved_duplicate = duplicate_bbl_sale
        if "property_id_source" in out.columns:
            unresolved_duplicate = unresolved_duplicate & (out["property_id_source"].astype("string") == "bbl_only")
        add_reason(unresolved_duplicate, "possible_unit_identity_collision_review")

    if "days_since_last_sale" in out.columns:
        days_since_last_sale = pd.to_numeric(out["days_since_last_sale"], errors="coerce")
        add_reason((days_since_last_sale >= 0) & (days_since_last_sale <= 30), "rapid_resale_review")

    if "sqft_imputed" in out.columns:
        add_reason(_truthy_series(out["sqft_imputed"]), "etl_sqft_imputed_review")
    if "year_built_imputed" in out.columns:
        add_reason(_truthy_series(out["year_built_imputed"]), "etl_year_built_imputed_review")

    if "h3_index" in out.columns:
        h3_missing = out["h3_index"].isna() | (out["h3_index"].astype("string").str.strip() == "")
        add_reason(h3_missing, "missing_h3_index", exclude_training=True)
    if "property_id" in out.columns:
        property_id_missing = out["property_id"].isna() | (out["property_id"].astype("string").str.strip() == "")
        add_reason(property_id_missing, "missing_property_id_review")

    statuses = []
    joined_reasons = []
    for idx, row_reasons in enumerate(reasons):
        if exclude[idx]:
            statuses.append("exclude_training")
        elif row_reasons:
            statuses.append("review")
        else:
            statuses.append("valid_training_sale")
        joined_reasons.append(";".join(sorted(set(row_reasons))) if row_reasons else "none")

    out["sale_validity_status"] = statuses
    out["sale_validity_reasons"] = joined_reasons
    return out


def filter_training_eligible_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Drop only records that are explicitly unsafe for training."""
    if "sale_validity_status" not in df.columns:
        df = add_sale_validity_labels(df)
    return df[df["sale_validity_status"] != "exclude_training"].copy()


def restore_model_critical_missingness(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Revert ETL-level imputation for model-critical fields so model fitting owns imputation.

    The ETL CSV keeps imputed values for product/data-quality reporting. For modeling,
    those values are reset to missing and the sklearn pipeline fits imputation on the
    training split only.
    """
    out = df.copy()
    report: dict[str, Any] = {"fields": {}, "policy": "reset_etl_imputed_model_critical_values_to_missing"}

    for field, flag_col in MODEL_CRITICAL_IMPUTED_FIELDS.items():
        if field not in out.columns:
            continue
        values = pd.to_numeric(out[field], errors="coerce")
        invalid_mask = values.isna()
        if field in {"gross_square_feet", "year_built"}:
            invalid_mask = invalid_mask | (values <= 0)
        flag_mask = _truthy_series(out[flag_col]) if flag_col in out.columns else pd.Series(False, index=out.index)
        reset_mask = invalid_mask | flag_mask
        out.loc[reset_mask, field] = np.nan
        report["fields"][field] = {
            "flag_column": flag_col if flag_col in out.columns else "not_present",
            "reset_to_missing_rows": int(reset_mask.sum()),
            "original_missing_or_invalid_rows": int(invalid_mask.sum()),
            "etl_imputed_rows": int(flag_mask.sum()),
        }

    if "year_built" in out.columns and "building_age" in out.columns:
        year_missing = pd.to_numeric(out["year_built"], errors="coerce").isna()
        out.loc[year_missing, "building_age"] = np.nan
        report["fields"]["building_age"] = {
            "policy": "reset_when_year_built_missing_after_etl_imputation_neutralization",
            "reset_to_missing_rows": int(year_missing.sum()),
        }

    return out, report


def add_asof_local_market_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_h3_prior_count: int = 3,
) -> AsOfFeatureResult:
    """Add strict as-of local H3 market features to train and holdout frames."""
    train = train_df.copy()
    test = test_df.copy()
    train_features = _compute_asof_features(train, train, min_h3_prior_count=min_h3_prior_count)
    test_features = _compute_asof_features(test, train, min_h3_prior_count=min_h3_prior_count)

    for column in LOCAL_MARKET_FEATURES:
        train[column] = train_features[column].reindex(train.index)
        test[column] = test_features[column].reindex(test.index)

    manifest = {
        "feature_names": LOCAL_MARKET_FEATURES,
        "as_of_policy": "strictly_less_than_row_sale_date",
        "reference_policy": {
            "train_rows": "expanding historical rows from the training split only",
            "test_rows": "training split rows only; holdout targets are never used",
        },
        "min_h3_prior_count_for_h3_median": int(min_h3_prior_count),
        "train_missing_rates": _missing_rates(train, LOCAL_MARKET_FEATURES),
        "test_missing_rates": _missing_rates(test, LOCAL_MARKET_FEATURES),
    }
    return AsOfFeatureResult(train_df=train, test_df=test, manifest=manifest)


def build_split_manifest_frame(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Create row-level split materialization for same-row challenger/champion checks."""
    train = _split_rows(train_df, "train")
    test = _split_rows(test_df, "test")
    manifest = pd.concat([train, test], ignore_index=True)
    return manifest.sort_values(["split", "sale_date", "row_id"]).reset_index(drop=True)


def split_manifest_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    """Summarize a row-level split manifest without losing row-hash auditability."""
    train_ids = manifest.loc[manifest["split"] == "train", "row_id"].astype(str).tolist()
    test_ids = manifest.loc[manifest["split"] == "test", "row_id"].astype(str).tolist()
    all_ids = manifest["row_id"].astype(str).tolist()
    return {
        "row_count": int(len(manifest)),
        "train_rows": int(len(train_ids)),
        "test_rows": int(len(test_ids)),
        "train_row_ids_sha256": _hash_tokens(train_ids),
        "test_row_ids_sha256": _hash_tokens(test_ids),
        "all_row_ids_sha256": _hash_tokens(all_ids),
        "duplicate_row_ids": int(manifest["row_id"].duplicated().sum()),
    }


def sale_validity_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize sale-validity labels for governance artifacts."""
    if "sale_validity_status" not in df.columns:
        return {"status": "not_generated"}
    status_counts = df["sale_validity_status"].value_counts(dropna=False).to_dict()
    reason_counts: dict[str, int] = defaultdict(int)
    if "sale_validity_reasons" in df.columns:
        for raw in df["sale_validity_reasons"].fillna("none").astype(str):
            for reason in raw.split(";"):
                token = reason.strip() or "none"
                reason_counts[token] += 1
    return {
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def feature_availability_manifest(feature_columns: list[str]) -> dict[str, Any]:
    """Document point-in-time feature availability assumptions for pre-comps readiness."""
    return {
        "feature_count": int(len(feature_columns)),
        "features": {
            feature: PRE_COMPS_FEATURE_AVAILABILITY.get(
                feature,
                {
                    "source": "inference_available_public_or_derived_field",
                    "as_of_policy": "covered by model feature contract and forbidden target-derived allowlist",
                },
            )
            for feature in feature_columns
        },
    }


def _compute_asof_features(
    target_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    min_h3_prior_count: int,
) -> pd.DataFrame:
    output = pd.DataFrame(index=target_df.index, columns=LOCAL_MARKET_FEATURES, dtype="float64")
    if target_df.empty:
        return output

    target = target_df.copy()
    reference = _eligible_reference_sales(reference_df)
    target[DATE_COL] = pd.to_datetime(target[DATE_COL], errors="coerce")
    reference[DATE_COL] = pd.to_datetime(reference[DATE_COL], errors="coerce")
    target = target.sort_values(DATE_COL)
    reference = reference.sort_values(DATE_COL)

    h3_prices: dict[str, list[float]] = defaultdict(list)
    h3_ppsf: dict[str, list[float]] = defaultdict(list)
    global_prices: list[float] = []
    global_ppsf: list[float] = []
    ref_records = list(reference[[DATE_COL, "h3_index", TARGET_COL, "gross_square_feet"]].itertuples(index=False, name=None))
    ref_idx = 0

    for current_date, date_rows in target.groupby(DATE_COL, sort=True):
        if pd.isna(current_date):
            continue
        while ref_idx < len(ref_records):
            ref_date, h3_index, sale_price, sqft = ref_records[ref_idx]
            if pd.isna(ref_date) or ref_date >= current_date:
                break
            price = _finite_positive(sale_price)
            h3_key = _h3_key(h3_index)
            if price is not None and h3_key:
                h3_prices[h3_key].append(price)
                global_prices.append(price)
                sqft_value = _finite_positive(sqft)
                if sqft_value is not None:
                    ppsf = price / sqft_value
                    if np.isfinite(ppsf) and ppsf > 0:
                        h3_ppsf[h3_key].append(float(ppsf))
                        global_ppsf.append(float(ppsf))
            ref_idx += 1

        global_price_median = float(np.median(global_prices)) if global_prices else np.nan
        global_ppsf_median = float(np.median(global_ppsf)) if global_ppsf else np.nan
        for idx, row in date_rows.iterrows():
            h3_key = _h3_key(row.get("h3_index"))
            prior_prices = h3_prices.get(h3_key, []) if h3_key else []
            prior_ppsf = h3_ppsf.get(h3_key, []) if h3_key else []
            output.loc[idx, "h3_prior_sale_count"] = float(len(prior_prices))
            output.loc[idx, "h3_prior_median_price"] = (
                float(np.median(prior_prices)) if len(prior_prices) >= min_h3_prior_count else global_price_median
            )
            output.loc[idx, "h3_prior_median_ppsf"] = (
                float(np.median(prior_ppsf)) if len(prior_ppsf) >= min_h3_prior_count else global_ppsf_median
            )

    return output


def _eligible_reference_sales(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sale_validity_status" in out.columns:
        out = out[out["sale_validity_status"] == "valid_training_sale"].copy()
    required = [DATE_COL, "h3_index", TARGET_COL]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Cannot build as-of local features; missing columns: {missing}")
    if "gross_square_feet" not in out.columns:
        out["gross_square_feet"] = np.nan
    return out.dropna(subset=[DATE_COL, "h3_index", TARGET_COL])


def _split_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "row_id": stable_row_ids(df),
            "split": split,
            "sale_date": pd.to_datetime(df.get(DATE_COL), errors="coerce").dt.strftime("%Y-%m-%d"),
            "property_id": df.get("property_id", pd.Series([""] * len(df), index=df.index)).astype("string").fillna(""),
            "h3_index": df.get("h3_index", pd.Series([""] * len(df), index=df.index)).astype("string").fillna(""),
        }
    )
    return out


def stable_row_ids(df: pd.DataFrame) -> pd.Series:
    """Build deterministic row IDs from stable property/sale identity fields."""
    parts = []
    for column in ["property_id", "bbl", "unit_identifier", DATE_COL, TARGET_COL, "h3_index"]:
        if column in df.columns:
            value = df[column]
        else:
            value = pd.Series([""] * len(df), index=df.index)
        if column == DATE_COL:
            value = pd.to_datetime(value, errors="coerce").dt.strftime("%Y-%m-%d")
        parts.append(value.astype("string").fillna(""))
    tokens = parts[0]
    for part in parts[1:]:
        tokens = tokens + "|" + part
    return tokens.apply(lambda token: hashlib.sha256(str(token).encode("utf-8")).hexdigest())


def _hash_tokens(tokens: list[str]) -> str:
    digest = hashlib.sha256()
    for token in sorted(tokens):
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


def _h3_key(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _missing_rates(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {column: float(df[column].isna().mean()) if column in df.columns and len(df) else 1.0 for column in columns}
