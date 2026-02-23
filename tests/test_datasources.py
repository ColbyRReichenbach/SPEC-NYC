from pathlib import Path

from src.datasources import get_datasource, list_datasources


def test_datasource_registry_lists_expected_names():
    names = list_datasources()
    assert "csv" in names
    assert "postgres" in names


def test_csv_datasource_extract(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("sale_date,sale_price,latitude,longitude,property_segment,gross_square_feet,year_built\n2025-01-01,500000,40.7,-73.9,WALKUP,900,1920\n", encoding="utf-8")
    ds = get_datasource("csv", input_path=csv_path)
    extract = ds.extract(limit=1)
    assert len(extract.df) == 1
