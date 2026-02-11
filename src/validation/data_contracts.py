"""Data contracts for ETL reliability checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd


DEFAULT_REQUIRED_COLUMNS = [
    "sale_date",
    "sale_price",
    "bbl",
    "latitude",
    "longitude",
    "borough",
    "block",
    "lot",
    "building_class",
    "building_class_category",
]

DEFAULT_NULL_THRESHOLDS = {
    "sale_date": 0.0,
    "sale_price": 0.0,
    "bbl": 0.0,
    "latitude": 0.0,
    "longitude": 0.0,
}

NYC_LAT_RANGE = (40.40, 41.10)
NYC_LON_RANGE = (-74.35, -73.45)


@dataclass
class ContractViolation:
    """Represents a failed data-contract check."""

    check: str
    message: str
    failed_rows: int = 0


@dataclass
class DataContractResult:
    """Aggregated result for all checks."""

    passed: bool
    row_count: int
    checked_at: str
    violations: List[ContractViolation]


class DataContractError(ValueError):
    """Raised when one or more contract checks fail."""


def validate_data_contracts(
    df: pd.DataFrame,
    *,
    required_columns: Optional[Iterable[str]] = None,
    null_thresholds: Optional[dict[str, float]] = None,
    max_freshness_days: int = 730,
    min_sale_price: int = 10_000,
    expected_bbl_length: int = 10,
    raise_on_error: bool = True,
) -> DataContractResult:
    """
    Execute ETL data-contract checks and optionally raise on failure.

    Checks:
    - required schema columns
    - sale_date freshness
    - null thresholds
    - domain constraints (sale_price, bbl length, coordinate bounds, borough bounds)
    """
    req_cols = list(required_columns or DEFAULT_REQUIRED_COLUMNS)
    thresholds = null_thresholds or DEFAULT_NULL_THRESHOLDS

    violations: List[ContractViolation] = []
    violations.extend(_check_required_columns(df, req_cols))
    violations.extend(_check_freshness(df, max_freshness_days=max_freshness_days))
    violations.extend(_check_null_thresholds(df, thresholds))
    violations.extend(
        _check_domain_constraints(
            df,
            min_sale_price=min_sale_price,
            expected_bbl_length=expected_bbl_length,
        )
    )

    result = DataContractResult(
        passed=len(violations) == 0,
        row_count=len(df),
        checked_at=datetime.utcnow().isoformat(),
        violations=violations,
    )

    if raise_on_error and not result.passed:
        raise DataContractError(format_contract_violations(result.violations))

    return result


def format_contract_violations(violations: List[ContractViolation]) -> str:
    """Format violations into a single error message."""
    lines = ["Data contract validation failed:"]
    for v in violations:
        lines.append(f"- [{v.check}] {v.message} (failed_rows={v.failed_rows})")
    return "\n".join(lines)


def _check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> List[ContractViolation]:
    missing = [col for col in required_columns if col not in df.columns]
    if not missing:
        return []
    return [
        ContractViolation(
            check="required_columns",
            message=f"Missing required columns: {', '.join(missing)}",
            failed_rows=len(df),
        )
    ]


def _check_freshness(df: pd.DataFrame, max_freshness_days: int) -> List[ContractViolation]:
    if "sale_date" not in df.columns:
        return []

    sale_dates = pd.to_datetime(df["sale_date"], errors="coerce")
    if sale_dates.notna().sum() == 0:
        return [
            ContractViolation(
                check="freshness",
                message="No valid sale_date values found",
                failed_rows=len(df),
            )
        ]

    latest_date = pd.Timestamp(sale_dates.max())
    if latest_date.tzinfo is not None:
        latest_date = latest_date.tz_convert("UTC").tz_localize(None)
    latest_date = latest_date.normalize()

    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is not None:
        now_utc = now_utc.tz_convert("UTC").tz_localize(None)
    age_days = (now_utc.normalize() - latest_date).days
    if age_days <= max_freshness_days:
        return []

    return [
        ContractViolation(
            check="freshness",
            message=f"Latest sale_date {latest_date.date()} is stale ({age_days} days old, max {max_freshness_days})",
            failed_rows=len(df),
        )
    ]


def _check_null_thresholds(df: pd.DataFrame, null_thresholds: dict[str, float]) -> List[ContractViolation]:
    violations: List[ContractViolation] = []
    for col, threshold in null_thresholds.items():
        if col not in df.columns:
            continue
        null_ratio = float(df[col].isna().mean())
        if null_ratio > threshold:
            violations.append(
                ContractViolation(
                    check="null_threshold",
                    message=f"{col} null ratio {null_ratio:.4f} exceeds threshold {threshold:.4f}",
                    failed_rows=int(df[col].isna().sum()),
                )
            )
    return violations


def _check_domain_constraints(
    df: pd.DataFrame,
    *,
    min_sale_price: int,
    expected_bbl_length: int,
) -> List[ContractViolation]:
    violations: List[ContractViolation] = []

    if "sale_price" in df.columns:
        sale_price = pd.to_numeric(df["sale_price"], errors="coerce")
        bad_sale_price = sale_price < min_sale_price
        if bad_sale_price.any():
            violations.append(
                ContractViolation(
                    check="domain_sale_price",
                    message=f"sale_price below {min_sale_price}",
                    failed_rows=int(bad_sale_price.sum()),
                )
            )

    if "bbl" in df.columns:
        bbl_numeric = pd.to_numeric(df["bbl"], errors="coerce")
        bbl_text = bbl_numeric.astype("Int64").astype(str)
        bbl_len = bbl_text.str.len()
        invalid_bbl = bbl_numeric.isna() | (bbl_len != expected_bbl_length)
        if invalid_bbl.any():
            violations.append(
                ContractViolation(
                    check="domain_bbl",
                    message=f"bbl must be numeric with {expected_bbl_length} digits",
                    failed_rows=int(invalid_bbl.sum()),
                )
            )

    if {"latitude", "longitude"}.issubset(df.columns):
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        bad_coords = (
            lat.isna()
            | lon.isna()
            | (lat < NYC_LAT_RANGE[0])
            | (lat > NYC_LAT_RANGE[1])
            | (lon < NYC_LON_RANGE[0])
            | (lon > NYC_LON_RANGE[1])
        )
        if bad_coords.any():
            violations.append(
                ContractViolation(
                    check="domain_coordinates",
                    message="coordinates outside expected NYC bounds",
                    failed_rows=int(bad_coords.sum()),
                )
            )

    if "borough" in df.columns:
        borough = pd.to_numeric(df["borough"], errors="coerce")
        bad_borough = borough.isna() | (~borough.isin([1, 2, 3, 4, 5]))
        if bad_borough.any():
            violations.append(
                ContractViolation(
                    check="domain_borough",
                    message="borough must be one of 1..5",
                    failed_rows=int(bad_borough.sum()),
                )
            )

    return violations
