from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import CANONICAL_OPTIONAL_COLUMNS, CANONICAL_REQUIRED_COLUMNS

MIN_MARKET_SALE_PRICE = 10_000


def to_canonical(raw_df: pd.DataFrame, mapping_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    mapping = _load_mapping(mapping_path)
    rename_map = mapping.get("rename", {})
    defaults = mapping.get("defaults", {})

    df = raw_df.copy()
    if rename_map:
        df = df.rename(columns=rename_map)

    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    all_cols = [*CANONICAL_REQUIRED_COLUMNS, *CANONICAL_OPTIONAL_COLUMNS]
    for col in all_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    for numeric_col in ("sale_price", "latitude", "longitude", "gross_square_feet"):
        df[numeric_col] = _coerce_numeric(df[numeric_col])
    # Annualized sales feeds encode non-market transactions as 0/low values; treat as missing so
    # canonical contracts focus on parseability and downstream clean_data can apply market filters.
    df.loc[df["sale_price"] < MIN_MARKET_SALE_PRICE, "sale_price"] = pd.NA
    df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce").fillna(0).astype("int64")

    report = {
        "mapping_path": str(mapping_path) if mapping_path else None,
        "rename_count": len(rename_map),
        "defaults_count": len(defaults),
        "input_columns": len(raw_df.columns),
        "canonical_columns": len(df.columns),
    }
    return df, report


def write_mapping_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _load_mapping(mapping_path: Path | None) -> dict:
    if mapping_path is None:
        return {}

    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    text = mapping_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ValueError("Mapping file must be JSON unless pyyaml is installed.") from exc
        data = yaml.safe_load(text)
        return data or {}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA, "n/a": pd.NA, "na": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")
