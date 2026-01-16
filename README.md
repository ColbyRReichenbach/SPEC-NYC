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
| Rolling Sales | [NYC Finance](https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page) | Transaction prices |
| PLUTO | [NYC Planning](https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page) | Building characteristics |

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
