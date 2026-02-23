from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import pandas as pd


@dataclass
class RawExtract:
    """Source-native extract with lineage metadata."""

    df: pd.DataFrame
    extracted_at: datetime
    source_id: str
    source_fingerprint: str
    metadata: dict[str, Any]


class DataSource(Protocol):
    """Pluggable data source contract for extraction."""

    def extract(self, *, start_date: str | None = None, end_date: str | None = None, limit: int | None = None) -> RawExtract:
        ...

    def describe(self) -> dict[str, Any]:
        ...
