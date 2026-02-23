from pathlib import Path

import pandas as pd

from src.canonical import to_canonical, validate_canonical_contracts


def test_to_canonical_with_mapping_yaml():
    df = pd.DataFrame(
        {
            "close_date": ["2025-01-01"],
            "close_price": [750000],
            "lat": [40.72],
            "lon": [-73.99],
            "segment_hint": ["WALKUP"],
            "sqft": [900],
            "built_year": [1920],
        }
    )
    out, report = to_canonical(df, Path("src/datasources/mappings/internship_example.yaml"))
    assert "sale_price" in out.columns
    assert report["rename_count"] > 0
    result = validate_canonical_contracts(out, raise_on_error=False)
    assert result.passed


def test_to_canonical_normalizes_low_sale_price_and_numeric_strings():
    df = pd.DataFrame(
        {
            "sale_date": ["2025-01-01", "2025-01-02"],
            "sale_price": ["0", "9,500"],
            "latitude": ["40.72", "40.73"],
            "longitude": ["-73.99", "-73.98"],
            "property_segment": ["WALKUP", "WALKUP"],
            "gross_square_feet": ["1,100", "980"],
            "year_built": ["1920", "1930"],
        }
    )

    out, _ = to_canonical(df)
    assert out["gross_square_feet"].tolist() == [1100.0, 980.0]
    assert out["sale_price"].isna().all()
    result = validate_canonical_contracts(out, raise_on_error=False)
    assert result.passed
