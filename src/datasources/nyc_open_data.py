from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from src.connectors import download_annualized_sales

from .base import RawExtract


@dataclass
class NYCOpenDataSource:
    start_year: int = 2019
    end_year: int = 2025
    cache: bool = True

    def extract(self, *, start_date: str | None = None, end_date: str | None = None, limit: int | None = None) -> RawExtract:
        df = download_annualized_sales(start_year=self.start_year, end_year=self.end_year, cache=self.cache)
        if limit is not None:
            df = df.head(limit).copy()
        fingerprint = hashlib.sha256(f"nyc_open_data:{self.start_year}:{self.end_year}:{len(df)}".encode()).hexdigest()[:16]
        return RawExtract(
            df=df,
            extracted_at=datetime.now(timezone.utc),
            source_id="nyc_open_data",
            source_fingerprint=fingerprint,
            metadata={"start_date": start_date, "end_date": end_date, "start_year": self.start_year, "end_year": self.end_year},
        )

    def describe(self) -> dict[str, object]:
        return {"name": "nyc_open_data", "start_year": self.start_year, "end_year": self.end_year, "cache": self.cache}
