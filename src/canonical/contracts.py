from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.validation.data_contracts import DataContractError, DataContractResult, ContractViolation
from .schema import CANONICAL_REQUIRED_COLUMNS


def validate_canonical_contracts(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str] = CANONICAL_REQUIRED_COLUMNS,
    min_sale_price: int = 10_000,
    raise_on_error: bool = True,
) -> DataContractResult:
    violations: list[ContractViolation] = []
    required = list(required_columns)
    missing = [col for col in required if col not in df.columns]
    if missing:
        violations.append(ContractViolation("required_columns", f"Missing canonical columns: {', '.join(missing)}", len(df)))

    if "sale_price" in df.columns:
        price = pd.to_numeric(df["sale_price"], errors="coerce")
        bad = price < min_sale_price
        if bad.any():
            violations.append(ContractViolation("domain_sale_price", f"sale_price below {min_sale_price}", int(bad.sum())))

    if "sale_date" in df.columns and pd.to_datetime(df["sale_date"], errors="coerce").notna().sum() == 0:
        violations.append(ContractViolation("sale_date", "No parseable sale_date values", len(df)))

    result = DataContractResult(
        passed=len(violations) == 0,
        row_count=len(df),
        checked_at=pd.Timestamp.utcnow().isoformat(),
        violations=violations,
    )
    if raise_on_error and not result.passed:
        from src.validation.data_contracts import format_contract_violations

        raise DataContractError(format_contract_violations(result.violations))
    return result
