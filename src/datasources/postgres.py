from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine

from .base import RawExtract


@dataclass
class PostgresDataSource:
    database_url: str
    query: str

    def extract(self, *, start_date: str | None = None, end_date: str | None = None, limit: int | None = None) -> RawExtract:
        sql = self.query
        if limit is not None:
            sql = f"SELECT * FROM ({self.query}) q LIMIT {int(limit)}"
        engine = create_engine(self.database_url)
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn)
        fingerprint = hashlib.sha256(f"postgres:{sql}:{len(df)}".encode()).hexdigest()[:16]
        return RawExtract(
            df=df,
            extracted_at=datetime.now(timezone.utc),
            source_id="postgres",
            source_fingerprint=fingerprint,
            metadata={"start_date": start_date, "end_date": end_date},
        )

    def describe(self) -> dict[str, object]:
        return {"name": "postgres", "query": self.query[:120]}
