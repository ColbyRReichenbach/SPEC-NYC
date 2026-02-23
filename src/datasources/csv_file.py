from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .base import RawExtract


@dataclass
class CsvFileDataSource:
    path: Path

    def extract(self, *, start_date: str | None = None, end_date: str | None = None, limit: int | None = None) -> RawExtract:
        df = pd.read_csv(self.path, low_memory=False)
        if limit is not None:
            df = df.head(limit).copy()
        fingerprint = hashlib.sha256(f"{self.path.resolve()}:{self.path.stat().st_mtime_ns}:{len(df)}".encode()).hexdigest()[:16]
        return RawExtract(
            df=df,
            extracted_at=datetime.now(timezone.utc),
            source_id="csv_file",
            source_fingerprint=fingerprint,
            metadata={"path": str(self.path), "start_date": start_date, "end_date": end_date},
        )

    def describe(self) -> dict[str, Any]:
        return {"name": "csv_file", "path": str(self.path)}
