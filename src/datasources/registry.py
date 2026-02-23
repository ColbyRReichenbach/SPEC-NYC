from __future__ import annotations

import json
from pathlib import Path

from config.settings import DATABASE_URL

from .base import DataSource
from .csv_file import CsvFileDataSource
from .nyc_open_data import NYCOpenDataSource
from .postgres import PostgresDataSource


def get_datasource(name: str, *, input_path: Path | None = None, config_path: Path | None = None) -> DataSource:
    normalized = name.strip().lower()
    config = _load_config(config_path)

    if normalized == "nyc_open_data":
        return NYCOpenDataSource(
            start_year=int(config.get("start_year", 2019)),
            end_year=int(config.get("end_year", 2025)),
            cache=bool(config.get("cache", True)),
        )
    if normalized == "csv":
        if input_path is None:
            raise ValueError("csv datasource requires input_path")
        return CsvFileDataSource(path=input_path)
    if normalized == "postgres":
        query = str(config.get("query", "SELECT * FROM sales"))
        db_url = str(config.get("database_url") or DATABASE_URL)
        return PostgresDataSource(database_url=db_url, query=query)
    raise ValueError(f"Unsupported data source: {name}")


def list_datasources() -> list[str]:
    return ["nyc_open_data", "csv", "postgres"]


def _load_config(config_path: Path | None) -> dict:
    if not config_path:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Datasource config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))
