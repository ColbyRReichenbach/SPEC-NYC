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

## Definition of Done (DoD) Gates

Each phase is complete only when its gate is green with evidence artifacts:

- **Gate A (Data Foundation)**:
  - ETL data contracts pass with zero critical violations
  - `tests/test_etl.py` passes
  - ETL run artifacts exist in `reports/data/`
- **Gate B (Model Baseline)**:
  - Reproducible training run produces model + metrics artifacts
  - Overall and per-segment scorecards generated
- **Gate C (Product Experience)**:
  - Dashboard loads live KPI/valuation/explainability outputs
  - Core user journey works end-to-end (data -> prediction -> explanation)
- **Gate D (Operations)**:
  - MLflow tracking wired for params/metrics/artifacts/tags
  - Drift + performance monitor scripts generate outputs
- **Gate E (Release)**:
  - Validator readiness report is all-green
  - Release checklist complete and tag created

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

### 1.5 Data Cleaning & Enrichment ✅

**Filtering Rules:**
- [x] Filter `sale_price < 10000` (family transfers, non-market)
- [x] Filter non-residential building classes
- [x] Filter missing coordinates (cannot map)
- [x] Filter missing BBL (no property identifier)

**Property Identification (NEW):**
- [x] Create `property_id` = BBL + apartment_number (for condos/co-ops)
- [x] Keep ALL legitimate resales (same property, different dates)
- [x] Remove only TRUE duplicates (same property_id + date + price)

**Sales History Enrichment (NEW):**
- [x] `sale_sequence`: 1st, 2nd, 3rd sale of this property in dataset
- [x] `is_latest_sale`: Boolean flag for most recent sale per property
- [x] `previous_sale_price`: For appreciation calculation
- [x] `previous_sale_date`: For holding period analysis
- [x] `price_change_pct`: (current - previous) / previous

**Missing Value Imputation:**
- [x] `gross_square_feet`: Hierarchical median (neighborhood+class → borough+class → class → citywide)
- [x] `year_built`: Hierarchical median (neighborhood → borough → citywide)
- [x] Track all imputations with `_imputed` flags
- [x] Document in `docs/DATA_QUALITY_LOG.md`

**ETL Results (2025-01-17)**:
- Raw records: 498,666
- Final cleaned records: 294,837
- Unique properties: 206,426
- Properties with resales: 88,411
- Segments: SINGLE_FAMILY (43.5%), ELEVATOR (42.4%), WALKUP (8.8%), SMALL_MULTI (5.3%)

### 1.6 Property Segmentation (NEW) ✅

**Segment Classification:**
```
Segment          | Building Classes | Est. Records | Key Characteristics
─────────────────┼──────────────────┼──────────────┼────────────────────
SINGLE_FAMILY    | 01, 02, 03       | 128,266      | Houses, standalone
WALKUP           | 07, 09, 12       | 25,944       | No elevator apts
ELEVATOR         | 08, 10, 13       | 125,072      | Doorman buildings
SMALL_MULTI      | 14, 15, 16, 17   | 15,555       | 2-10 unit buildings
```

- [x] Add `property_segment` column based on building_class
- [x] Add `price_tier` column (within-segment quartile: entry/core/premium/luxury)
- [x] Validate segment distribution

### 1.7 Feature Engineering ✅

**Spatial Features:**
- [x] `distance_to_center_km` - Distance to Manhattan (40.7831, -73.9712)
- [x] `h3_index` - Uber H3 hex at resolution 8
- [x] `h3_price_lag` - Median price in neighboring hexes (computed at training time)

**Time Features:**
- [x] `sale_month`, `sale_quarter` - Seasonality
- [ ] `days_on_market` - If available from listing data

**Derived Features:**
- [x] `building_age` = current_year - year_built
- [x] `price_per_sqft` = sale_price / gross_square_feet (for comps, not training)

### 1.8 Model Training (Global + Segment Features)

**V1.0 Approach: Single Global Model with Segment Features**

Per industry research (Zillow, Redfin), we start with ONE model that learns segments implicitly:

```python
NUMERIC_FEATURES = [
    'gross_square_feet', 'year_built', 'building_age',
    'residential_units', 'total_units',
    'distance_to_center_km', 'h3_price_lag'
]

CATEGORICAL_FEATURES = [
    'borough',              # 5 categories
    'building_class',       # ~48 categories
    'property_segment',     # 4 categories (NEW)
    'price_tier',           # 4 categories (NEW)
    'neighborhood',         # ~250 categories (or use H3)
]
```

- [x] Train XGBoost with Optuna tuning path (trials configurable via CLI)
- [x] Evaluate OVERALL: PPE10, MdAPE, R²
- [x] Evaluate BY SEGMENT: PPE10 per property_segment
- [x] If segment variance >15%, flag for V2.0 segment-specific models
- [x] Save model to `models/` directory
- [x] Log metrics to MLflow

### 1.9 Dashboard

- [x] Port `app.py` from SF project
- [x] Update for NYC:
  - Map centered on Manhattan
  - Borough and segment filters
  - Show `is_latest_sale` properties by default
  - Click for full price history
- [x] Display:
  - Property map with valuation status (color by over/under valued)
  - SHAP waterfall chart
  - Property details + sales history
- [x] Test: Application runs without errors

### 1.10 V1.0 Deliverables Checklist

- [ ] Docker Compose works (`docker-compose up`)
- [x] PostgreSQL contains cleaned, segmented NYC data
- [x] All imputation documented in DATA_QUALITY_LOG.md
- [x] Global model achieves ≥70% PPE10 overall
- [x] Per-segment PPE10 logged and analyzed
- [x] SHAP explanations display correctly
- [x] Map shows properties with price history on click
- [x] README documents data sources, segmentation, and metrics
- [x] Git tag: `v1.0`

---

## Phase 2: Segment-Specific Models (V2.0)

### 2.1 Segment Performance Analysis

If V1.0 shows >15% PPE10 variance between segments, train segment-specific models:

```
┌─────────────────────────────────────────────────────────────┐
│                     PROPERTY ROUTER                          │
│  (Route based on property_segment + price_tier)             │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ SINGLE_FAMILY │     │   ELEVATOR    │     │    WALKUP     │
│    MODEL      │     │    MODEL      │     │    MODEL      │
└───────────────┘     └───────────────┘     └───────────────┘
```

- [ ] Analyze V1.0 per-segment performance
- [ ] Identify segments with poor accuracy (PPE10 <65%)
- [ ] Train dedicated model for underperforming segments
- [ ] Compare: segment model vs global model per segment

### 2.2 Quantile Regression (Uncertainty)

- [ ] Train quantile models for confidence intervals:
  - `model_q10`: 10th percentile (lower bound)
  - `model_q50`: 50th percentile (point estimate)
  - `model_q90`: 90th percentile (upper bound)
- [ ] Use `objective='reg:quantileerror'` with `quantile_alpha`
- [ ] Prediction output:
```python
{
    "point_estimate": 1_250_000,
    "confidence_80": [1_050_000, 1_480_000],
    "segment": "ELEVATOR",
    "model_used": "segment_elevator_v2"
}
```

### 2.3 NYC Spatial Features

- [ ] Download MTA subway station data
- [ ] Add feature: `subway_distance_m` (distance to nearest station)
- [ ] Download FEMA flood zone data (if available)
- [ ] Add feature: `flood_zone` (boolean or category)
- [ ] Retrain models with new features
- [ ] Evaluate improvement by segment

### 2.4 Temporal Backtesting

- [ ] Create `src/backtest.py`:
  - Train on 2019-2022 data
  - Predict 2023-2024 sales
  - Compare predictions vs actuals BY SEGMENT
- [ ] Add dashboard section:
  - "Backtest Results" with segment breakdown
  - PPE10 on holdout years
  - Scatter plot: predicted vs actual (colored by segment)
- [ ] Target: ≥70% PPE10 on backtest for each segment

### 2.5 Comparable Sales

- [ ] Create `src/comps.py`:
  - Find 5 most similar sales within SAME SEGMENT
  - Similarity based on: neighborhood, sqft (±20%), building class
  - Within last 12 months
- [ ] Add dashboard section: "Comparable Sales"
- [ ] Display: address, sale price, date, sqft, similarity score
- [ ] Show appreciation trends for resold properties

### 2.6 Price History & Appreciation

- [ ] Dashboard widget: "Price History" for properties with multiple sales
- [ ] Calculate appreciation rate by neighborhood/segment
- [ ] "Hot Spots" map: neighborhoods with highest appreciation

### 2.7 V2.0 Deliverables Checklist

- [ ] Segment-specific models trained (if needed)
- [ ] Confidence intervals displayed in UI
- [ ] Subway distance feature integrated
- [ ] Backtesting by segment functional
- [ ] Comparable sales within segment
- [ ] Price history visible for resold properties
- [ ] PPE10 ≥75% per segment, MdAPE ≤8%
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
