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
