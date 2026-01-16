# S.P.E.C. NYC Implementation Plan

This document serves as the master workflow for building S.P.E.C.-NYC, a production-grade Automated Valuation Model using real NYC transaction data. Follow each phase sequentially. Check off tasks as completed.

---

## Project Overview

**Goal**: Transform the SF architectural demo into a business-ready tool using 1M+ real NYC transactions.

**Core Thesis**: Fast, explainable property valuation with quantified uncertainty.

**Target Metrics**:
- PPE10 (within ±10%): ≥75%
- MdAPE (median error): ≤8%
- R²: ≥0.75

---

## Data Sources

| Dataset | URL | Purpose |
|---------|-----|---------|
| NYC Rolling Sales | https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page | ~~Original source~~ Replaced by Annualized Sales API |
| PLUTO | https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page | ~~For coords~~ Not needed (Annualized has lat/lon) |
| **Annualized Sales** | `w2pb-icbu` API | **Primary source**: 498K records, 2019-2024 |
| MTA Subway Stations | https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f | Transit proximity |

> **📋 See `docs/DATA_SOURCES_DECISION_LOG.md`** for full rationale on why we use Annualized Sales.

**Join Key**: Borough-Block-Lot (BBL) identifier

---

## Phase 1: NYC Data Foundation (V1.0)

### 1.1 Project Setup

- [x] Create new repository `SPEC-NYC`
- [x] Initialize with Python .gitignore
- [x] Create directory structure:
```
SPEC-NYC/
├── .claude/
│   ├── agents/
│   └── commands/
├── config/
│   └── settings.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── src/
│   ├── __init__.py
│   ├── etl.py
│   ├── model.py
│   ├── spatial.py
│   └── connectors.py
├── app.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

### 1.2 Infrastructure Setup

- [x] Create `docker-compose.yml` with:
  - PostgreSQL 15 service (port 5432)
  - Streamlit app service (port 8501)
  - Shared network
- [x] Create `Dockerfile` for Streamlit app
- [x] Create `.env.example` with database credentials
- [ ] Test: `docker-compose up -d` starts both services

### 1.3 Database Schema ✅

- [x] Create `src/database.py` with SQLAlchemy models
- [x] Define tables (designed from actual Annualized Sales data):
  - `sales` - 30 columns including coordinates, sale_price, sqft, year_built, h3_index
  - `predictions` - predicted prices with uncertainty bounds
  - `model_performance` - track model metrics over versions
- [x] PostgreSQL running on port 5433 (Docker, avoids local postgres conflict)
- [x] All tables created with proper indexes

### 1.4 Data Ingestion ✅

> **Note**: Original plan called for Rolling Sales + PLUTO. We switched to Annualized Sales API 
> which includes coordinates. See `docs/DATA_SOURCES_DECISION_LOG.md` for rationale.

- [x] ~~Download NYC Rolling Sales~~ → Using Annualized Sales API instead
- [x] ~~Download PLUTO dataset~~ → Not needed (Annualized has lat/lon)
- [x] Create `src/connectors.py` with:
  - `download_annualized_sales(start_year, end_year)` → Downloads 498K records
  - `download_pluto_coordinates()` → Available if needed later
  - Rate limiting (2s delay), retry logic, caching
- [x] BBL available directly in Annualized Sales dataset

**Data Downloaded**:
- Records: 498,666 (2019-2024)
- Coverage: 97.4% have coordinates
- Cache: `data/raw/annualized_sales_2019_2025.csv`

### 1.5 Data Cleaning

- [ ] Create `src/etl.py` with cleaning rules:
  - Filter `sale_price == 0` (family transfers)
  - Filter `sale_price < 10000` (non-market)
  - Filter commercial building classes
  - Handle missing sqft (impute from building class median)
  - Remove duplicates (keep most recent per BBL)
- [ ] Log cleaning statistics (records removed, reasons)
- [ ] Save cleaned data to PostgreSQL

### 1.6 Feature Engineering

- [ ] Port `src/spatial.py` from SF project
- [ ] Update city center coordinates to Manhattan (40.7831, -73.9712)
- [ ] Compute features:
  - `distance_to_center_km`
  - `h3_index` (resolution 8)
  - `h3_price_lag` (median price in hex neighbors)
- [ ] Create feature matrix for model training

### 1.7 Model Training

- [ ] Port `src/model.py` from SF project
- [ ] Train XGBoost with Optuna (50 trials)
- [ ] Features:
  - sqft, year_built, units_total (from PLUTO)
  - building_class, borough (categorical)
  - h3_price_lag, distance_to_center_km (spatial)
- [ ] Evaluate: PPE10, MdAPE, R²
- [ ] Save model to `models/` directory
- [ ] Log to MLflow

### 1.8 Dashboard

- [ ] Port `app.py` from SF project
- [ ] Update for NYC:
  - Map centered on Manhattan
  - Borough filter (Manhattan, Brooklyn)
  - NYC-specific styling
- [ ] Display:
  - Property map with valuation status
  - SHAP waterfall chart
  - Basic property details
- [ ] Test: Application runs without errors

### 1.9 V1.0 Deliverables Checklist

- [ ] Docker Compose works (`docker-compose up`)
- [ ] PostgreSQL contains cleaned NYC data
- [ ] Model achieves ≥70% PPE10
- [ ] SHAP explanations display correctly
- [ ] Map shows Manhattan/Brooklyn properties
- [ ] README documents data sources and metrics
- [ ] Git tag: `v1.0`

---

## Phase 2: Probabilistic Intelligence (V2.0)

### 2.1 Quantile Regression

- [ ] Update `src/model.py` to train quantile models
- [ ] Train three XGBoost models:
  - `model_q10`: 10th percentile (lower bound)
  - `model_q50`: 50th percentile (point estimate)
  - `model_q90`: 90th percentile (upper bound)
- [ ] Use `objective='reg:quantileerror'` with `quantile_alpha`
- [ ] Prediction output format:
```python
{
    "point_estimate": 1_250_000,
    "confidence_80": [lower_q10, upper_q90],
}
```

### 2.2 NYC Spatial Features

- [ ] Download MTA subway station data
- [ ] Add feature: `subway_distance_m` (distance to nearest station)
- [ ] Download FEMA flood zone data (if available)
- [ ] Add feature: `flood_zone` (boolean or category)
- [ ] Retrain model with new features
- [ ] Evaluate improvement in metrics

### 2.3 Backtesting Dashboard

- [ ] Create `src/backtest.py`:
  - Train on 2018-2022 data
  - Predict 2023 sales
  - Compare predictions vs actuals
- [ ] Add dashboard section:
  - "2023 Backtest Results"
  - PPE10 on holdout year
  - Scatter plot: predicted vs actual
- [ ] Target: ≥70% PPE10 on backtest

### 2.4 Comparable Sales

- [ ] Create `src/comps.py`:
  - Find 5 most similar sales
  - Similarity based on: neighborhood, sqft (±20%), building class
  - Within last 12 months
- [ ] Add dashboard section: "Comparable Sales"
- [ ] Display: address, sale price, date, sqft, similarity score

### 2.5 Borough Expansion

- [ ] If Manhattan/Brooklyn PPE10 ≥75%:
  - Add Queens data
  - Add Bronx data
  - Add Staten Island data
- [ ] Retrain model on all 5 boroughs
- [ ] Evaluate per-borough accuracy
- [ ] Consider cascading model if accuracy drops significantly

### 2.6 V2.0 Deliverables Checklist

- [ ] Confidence intervals displayed in UI
- [ ] Subway distance feature integrated
- [ ] Backtesting dashboard functional
- [ ] Comparable sales table works
- [ ] All 5 boroughs supported (if feasible)
- [ ] PPE10 ≥75%, MdAPE ≤8%
- [ ] Git tag: `v2.0`

---

## Phase 3: Platform Shift (V3.0)

### 3.1 FastAPI Backend

- [ ] Create `api/` directory
- [ ] Create `api/main.py` with FastAPI app
- [ ] Implement endpoints:
  - `POST /valuation` - single property
  - `POST /explain` - SHAP breakdown
  - `POST /comps` - comparable sales
  - `POST /portfolio` - bulk upload (CSV)
  - `POST /report` - generate PDF
- [ ] Add Pydantic models for request/response validation
- [ ] Add error handling and logging

### 3.2 API Documentation

- [ ] Enable FastAPI auto-docs (`/docs`)
- [ ] Add example requests/responses
- [ ] Document rate limits (if any)

### 3.3 React Frontend

- [ ] Create `frontend/` directory
- [ ] Initialize Next.js app: `npx create-next-app@latest ./`
- [ ] Install dependencies:
  - `react-leaflet` or `react-map-gl` (mapping)
  - `recharts` or `plotly.js` (charts)
  - `tailwindcss` (styling)
- [ ] Create components:
  - `PropertyMap` - interactive map
  - `ValuationCard` - price + confidence interval
  - `ShapChart` - waterfall visualization
  - `CompsTable` - comparable sales
  - `PortfolioUpload` - CSV upload form

### 3.4 Frontend-Backend Integration

- [ ] Create API client in frontend
- [ ] Connect map clicks to `/valuation` endpoint
- [ ] Display SHAP chart from `/explain` response
- [ ] Implement portfolio upload flow

### 3.5 PDF Report Generation

- [ ] Install WeasyPrint or ReportLab
- [ ] Create report template with sections:
  - Header (address, date)
  - Valuation summary (price, confidence interval)
  - SHAP waterfall chart (as image)
  - Comparable sales table
  - Disclaimer
- [ ] `/report` endpoint returns PDF bytes
- [ ] Frontend download button

### 3.6 Portfolio Bulk Processing

- [ ] `/portfolio` accepts CSV with addresses
- [ ] Background processing (consider Celery for large files)
- [ ] Return results as downloadable CSV:
  - address, estimate, lower_ci, upper_ci, status
- [ ] Progress indicator in frontend

### 3.7 Docker Compose Update

- [ ] Add FastAPI service
- [ ] Add frontend service (nginx for production build)
- [ ] Update networking
- [ ] Environment variables for API URL

### 3.8 V3.0 Deliverables Checklist

- [ ] FastAPI backend with all endpoints
- [ ] React frontend with interactive map
- [ ] PDF report generation works
- [ ] Portfolio bulk upload functional
- [ ] Docker Compose runs all services
- [ ] API response time <500ms
- [ ] Git tag: `v3.0`

---

## Phase 4: Intelligence Layer (V4.0)

### 4.1 AI Investment Memos

- [ ] Port `src/oracle.py` from SF project
- [ ] Update system prompt for NYC market context
- [ ] Create `/memo` endpoint
- [ ] Integrate into frontend (expandable section)

### 4.2 NYC Market RAG

- [ ] Set up ChromaDB for vector storage
- [ ] Collect NYC market reports (PDFs):
  - Douglas Elliman quarterly reports
  - REBNY market studies
- [ ] Chunk and embed documents
- [ ] Update oracle to query vector store for context

### 4.3 Performance Optimization

- [ ] Add Redis caching for predictions
- [ ] Implement model warm-up on startup
- [ ] Optimize batch inference with vectorization
- [ ] Target: 100 properties/minute throughput

### 4.4 Final Polish

- [ ] Professional README with:
  - Demo GIF/video
  - Architecture diagram
  - Performance metrics
  - Quick start guide
- [ ] Create GitHub releases for all versions
- [ ] Record demo video (optional but recommended)

### 4.5 V4.0 Deliverables Checklist

- [ ] AI memos integrated
- [ ] NYC market context in RAG
- [ ] Redis caching implemented
- [ ] Performance benchmarks documented
- [ ] Professional README complete
- [ ] Git tag: `v4.0`

---

## Success Metrics Summary

| Phase | Metric | Target |
|-------|--------|--------|
| V1.0 | PPE10 | ≥70% |
| V1.0 | Data records | ≥50,000 |
| V2.0 | PPE10 | ≥75% |
| V2.0 | Backtest PPE10 | ≥70% |
| V2.0 | CI Calibration | 80% within 80% CI |
| V3.0 | API latency | <500ms |
| V3.0 | Bulk throughput | 100/min |
| V4.0 | AI memo generation | <5s |

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | PostgreSQL | Relational, scalable, industry standard |
| ML Framework | XGBoost | Fast, interpretable, proven |
| Explainability | SHAP | Industry standard for tree models |
| Backend | FastAPI | Modern, async, auto-docs |
| Frontend | React/Next.js | Industry standard, smooth UX |
| Deployment | Docker Compose | Local dev, easy to scale to K8s |

---

## Files to Port from SF Project

These files can be adapted from the existing SF codebase:

| File | Adapt | Notes |
|------|-------|-------|
| `config/settings.py` | Heavy | Update coordinates, paths |
| `src/model.py` | Medium | Add quantile regression |
| `src/spatial.py` | Medium | Update for NYC geography |
| `src/oracle.py` | Light | Update prompts for NYC |
| `src/ai_security.py` | None | Copy as-is |
| `app.py` | Heavy | Port to React eventually |

---

## Deferred Features (Out of Scope)

| Feature | Reason |
|---------|--------|
| User authentication | Portfolio project, not SaaS |
| Multi-city support | Prove NYC works first |
| Real-time data feeds | Monthly updates sufficient |
| Mobile app | Web is sufficient |
| USPAP compliance | Commercial requirement |

---

## Quick Reference Commands

```bash
# Start all services
docker-compose up -d

# Run ETL pipeline
python -m src.etl

# Train model
python -m src.model

# Run tests
pytest tests/

# Start Streamlit (dev mode)
streamlit run app.py

# Start FastAPI (dev mode)
uvicorn api.main:app --reload

# Build frontend
cd frontend && npm run build
```

---

## Notes for AI Agent

When working on this project:

1. **Always check the current phase** before starting work
2. **Complete tasks sequentially** within each phase
3. **Test after each major change** before moving on
4. **Commit frequently** with descriptive messages
5. **Update README** as features are added
6. **Log metrics** after model training

The goal is a portfolio piece that demonstrates:
- Data engineering (ETL from government APIs)
- ML engineering (XGBoost, SHAP, quantile regression)
- Full-stack development (FastAPI + React)
- Production thinking (Docker, caching, testing)
