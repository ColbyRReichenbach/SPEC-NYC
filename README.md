# S.P.E.C. Valuation Engine - NYC Edition

**Spatial · Predictive · Explainable · Conversational**

A production-grade Automated Valuation Model (AVM) for NYC residential real estate. Uses machine learning with quantified uncertainty and SHAP-based explainability.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600?style=flat)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?style=flat&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)

---

## 🚧 Status: V1.0 In Development

This project is being built following the [Implementation Plan](docs/NYC_IMPLEMENTATION_PLAN.md).

---

## Project Goals

| Goal | Description |
|------|-------------|
| **Real Data** | 1M+ actual NYC transactions (not simulated) |
| **Uncertainty** | Confidence intervals, not just point estimates |
| **Explainability** | SHAP-based feature attribution |
| **Production Ready** | PostgreSQL, Docker, API-first architecture |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/ColbyRReichenbach/SPEC-NYC.git
cd SPEC-NYC

# 2. Copy environment file
cp .env.example .env

# 3. Start services
docker-compose up -d

# 4. Access dashboard
open http://localhost:8501
```

---

## Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| Annualized Sales (`w2pb-icbu`) | [NYC Open Data](https://data.cityofnewyork.us/resource/w2pb-icbu) | Primary sales dataset (includes lat/lon + BBL) |
| MTA Subway Stations | [data.ny.gov](https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f) | Transit proximity features (planned) |

As of **February 10, 2026**, the Annualized Sales API range is `2016-01-01` to `2024-12-31`.

See `docs/DATA_SOURCES_DECISION_LOG.md` for source-selection rationale.

---

## Data Bootstrap (Real NYC Data)

```bash
# Canonical bootstrap (download + db + idempotent ETL load)
./scripts/bootstrap_data.sh
```

Useful variants:

```bash
# Dry-run ETL only (no DB load)
./scripts/bootstrap_data.sh --dry-run

# Reuse already-downloaded raw file and already-running DB
./scripts/bootstrap_data.sh --skip-download --skip-db-start

# Smaller pull window for quick validation
./scripts/bootstrap_data.sh --start-year 2024 --end-year 2024
```

Expected outputs:
- Raw file: `data/raw/annualized_sales_2019_2025.csv`
- ETL report: `reports/data/etl_run_YYYYMMDD.md`
- Postgres table: `sales` populated with cleaned/segmented rows

Optional:
- Set `SOCRATA_APP_TOKEN` in `.env` to improve API reliability on large pulls.

---

## V1 Baseline Snapshot

Data snapshot (latest ETL):
- Cleaned + loaded rows: `293,716`
- Date range: `2019-01-01` to `2024-12-31`
- Segments: `SINGLE_FAMILY`, `ELEVATOR`, `WALKUP`, `SMALL_MULTI`

Model baseline (`models/metrics_v1.json`):
- Overall PPE10: `87.5%`
- Overall MdAPE: `3.89%`
- Overall R2: `0.796`
- Segment PPE10: `ELEVATOR 88.37%`, `SINGLE_FAMILY 86.05%`, `SMALL_MULTI 100.00%`, `WALKUP 76.47%`

Artifacts:
- Metrics: `models/metrics_v1.json`
- Segment scorecard: `reports/model/segment_scorecard_v1.csv`
- SHAP plots: `reports/model/shap_summary_v1.png`, `reports/model/shap_waterfall_v1.png`

---

## Roadmap

- [ ] **V1.0**: NYC data pipeline, XGBoost model, Streamlit UI
- [ ] **V2.0**: Quantile regression, NYC spatial features, backtesting
- [ ] **V3.0**: FastAPI backend, React frontend, PDF reports
- [ ] **V4.0**: AI investment memos, performance optimization

See [Implementation Plan](docs/NYC_IMPLEMENTATION_PLAN.md) for detailed task breakdown.

---

## Related Project

This is the production-data version of the [S.P.E.C. Valuation Engine (SF)](https://github.com/ColbyRReichenbach/S.P.E.C-Valuation), which demonstrates the architecture using simulated data.

---

## Author

**Colby Reichenbach**  
[GitHub](https://github.com/ColbyRReichenbach) · [LinkedIn](https://linkedin.com/in/colbyreichenbach)

---

## License

MIT License - See [LICENSE](LICENSE) for details.
