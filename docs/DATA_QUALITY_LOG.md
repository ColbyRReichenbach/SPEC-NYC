# S.P.E.C. NYC Data Quality & Transformation Log

This document provides full transparency on all data quality issues, transformations, imputations, and decisions made during the ETL pipeline. Every modification to the raw data is documented here.

---

## Table of Contents

1. [Data Source](#data-source)
2. [Pipeline Overview](#pipeline-overview)
3. [Filtering Decisions](#filtering-decisions)
4. [Missing Value Analysis](#missing-value-analysis)
5. [Imputation Strategy](#imputation-strategy)
6. [Data Quality Metrics](#data-quality-metrics)
7. [Known Limitations](#known-limitations)
8. [Changelog](#changelog)

---

## Data Source

| Attribute | Value |
|-----------|-------|
| **Source** | NYC Annualized Sales API |
| **API Endpoint** | `w2pb-icbu` |
| **Date Range** | 2019-2024 |
| **Raw Records** | 498,666 |
| **Download Date** | 2026-02-10 |

---

## Pipeline Overview

```
Raw Data (498,666)
    │
    ├─► Filter: sale_price < $10,000 ──────► -162,207 records (family transfers, non-market)
    │
    ├─► Filter: Non-residential ───────────► -34,106 records (commercial only)
    │
    ├─► Filter: Missing coordinates ───────► -4,198 records (cannot map)
    │
    ├─► Filter: Missing/invalid BBL ───────► -2,238 records (missing or zero BBL)
    │
    ├─► Deduplicate (BBL + date + price) ──► -2,201 records (exact duplicates)
    │
    └─► Final Dataset: 293,716 records (58.9% retention)
```

---

## Filtering Decisions

### 1. Sale Price Threshold ($10,000)

**Decision**: Exclude all sales under $10,000

**Rationale**: 
- Sales at $0 are typically intra-family transfers, not market transactions
- Sales under $10,000 are non-arm's-length transactions (foreclosures, partial interests)
- Industry standard for NYC data (per NYC DOF guidelines)

**Impact**: -162,207 records (32.5%)

### 2. Residential Only

**Decision**: Keep only building class categories starting with residential prefixes:
- `01` ONE FAMILY DWELLINGS
- `02` TWO FAMILY DWELLINGS
- `03` THREE FAMILY DWELLINGS
- `07` RENTALS - WALKUP APARTMENTS
- `08` RENTALS - ELEVATOR APARTMENTS
- `09` COOPS - WALKUP APARTMENTS
- `10` COOPS - ELEVATOR APARTMENTS
- `12` CONDOS - WALKUP APARTMENTS
- `13` CONDOS - ELEVATOR APARTMENTS
- `14` RENTALS - 4-10 UNIT
- `15` CONDOS - 2-10 UNIT RESIDENTIAL
- `16` CONDOS - 2-10 UNIT WITH COMMERCIAL
- `17` CONDO COOPS

**Rationale**: Commercial properties have fundamentally different valuation drivers

**Impact**: -34,106 records (6.8%)

### 3. Coordinate Requirement

**Decision**: Exclude records without latitude/longitude

**Rationale**: Spatial features (H3 index, distance to center) are critical for model accuracy

**Impact**: -4,198 records (0.8%)

### 4. BBL Requirement

**Decision**: Exclude records without Borough-Block-Lot identifier

**Rationale**: BBL is the canonical NYC property identifier; required for deduplication

**Impact**: -2,238 records (0.4%)

---

## Missing Value Analysis

### Audit Date: 2026-02-11

| Feature | Total | NULL | Zero | Valid (>0) | Valid % | Root Cause |
|---------|-------|------|------|------------|---------|------------|
| `gross_square_feet` | 293,716 | 0 | 0 | 293,716 | **100.0%** | Missing values imputed hierarchically |
| `land_square_feet` | 293,716 | 133,928 | 18,950 | 140,838 | 47.9% | Not required for V1 model |
| `year_built` | 293,716 | 0 | 0 | 293,716 | 100.0% | Missing values imputed hierarchically |
| `residential_units` | 293,716 | 84,046 | 1,210 | 208,460 | 71.0% | Unit sales don't report building totals |
| `total_units` | 293,716 | 84,046 | 1,206 | 208,464 | 71.0% | Same as above |

### Root Cause Analysis: `gross_square_feet`

**Finding**: Co-ops and Condos represent 52% of our data but have ~100% missing sqft.

| Building Class Category | Missing Sqft | Total | Missing % |
|------------------------|--------------|-------|-----------|
| 10 COOPS - ELEVATOR APARTMENTS | 79,076 | 79,076 | 100% |
| 13 CONDOS - ELEVATOR APARTMENTS | 76,997 | 77,903 | 98.8% |
| 09 COOPS - WALKUP APARTMENTS | 15,677 | 15,677 | 100% |
| 04 TAX CLASS 1 CONDOS | 9,606 | 9,723 | 98.8% |
| 15 CONDOS - 2-10 UNIT RESIDENTIAL | 8,986 | 9,093 | 98.8% |

**Explanation**: 
- **Co-ops** are sold as shares in a corporation, not deeded property. The city records the sale but not the unit size.
- **Condos** sometimes have unit sqft, but NYC DOF does not require it. Only ~1.2% of condo sales include sqft.

**This is a known limitation of NYC public data, not an ETL bug.**

---

## Imputation Strategy

### Feature: `gross_square_feet`

**Status**: ✅ IMPLEMENTED

**Strategy**: Hierarchical Median Imputation

| Level | Approach | Description |
|-------|----------|-------------|
| 1 | Neighborhood + Building Class | Most granular, uses local comps |
| 2 | Borough + Building Class | Falls back if no local data |
| 3 | Building Class only | Citywide median for class |
| 4 | Citywide median | Final fallback (1,200 sqft default) |

**Tracking**:
- `sqft_imputed = True/False` - Was this value imputed?
- `sqft_imputation_level` - Which level was used?

**Justification**: 
- Preserves all data for model training
- Hierarchical approach reduces imputation error
- Median is robust to outliers (better than mean)
- Clearly flagged as imputed for downstream analysis

### Feature: `year_built`

**Status**: ✅ IMPLEMENTED

**Strategy**: Hierarchical Median Imputation
1. Neighborhood median
2. Borough median
3. Citywide median (default: 1960)

**Tracking**: `year_built_imputed = True/False`

### Feature: `residential_units` / `total_units`

**Status**: NOT IMPUTED

**Decision**: XGBoost handles NaN natively. These are supplementary features.

### Feature: `land_square_feet`

**Status**: NOT IMPUTED

**Decision**: Not a primary feature for unit-level valuation.

---

## Data Quality Metrics

### After ETL Pipeline (v1 - 2026-02-11)

| Metric | Value |
|--------|-------|
| Total records | 293,716 |
| Records with imputed sqft | 153,824 (52.4%) |
| Sqft imputation level distribution | neighborhood_class: 60,818; borough_class: 7,932; class_only: 9; citywide: 85,065 |
| Records with imputed year_built | 13,096 (4.5%) |
| Records with valid sqft | 293,716 (100.0%) |
| Records with valid year_built | 293,716 (100.0%) |
| Geographic coverage | 100% (all have lat/lon) |
| H3 index coverage | 100% |
| Borough coverage | All 5 boroughs |
| Date range | 2019-01-01 to 2024-12-31 |

---

## Known Limitations

1. **Co-op/Condo Square Footage**: NYC public data does not include unit-level sqft for co-ops and most condos. Imputed values should be interpreted with caution.

2. **Family Transfers Excluded**: Sales under $10K are excluded, which may undercount properties in distressed neighborhoods.

3. **Commercial Exclusion**: Mixed-use properties with commercial components are excluded if building class is commercial-primary.

4. **Temporal Bias**: 2020 sales may reflect COVID-related market distortions.

---

## Changelog

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2026-01-16 | 0.1 | Initial data download (498,666 records) | ETL Pipeline |
| 2026-01-17 | 0.2 | Filtering + cleaning + PostgreSQL load (290,996 records) | ETL Pipeline |
| 2026-01-17 | 0.3 | Added H3 spatial index and distance_to_center features | ETL Pipeline |
| 2026-01-17 | - | Created data quality documentation | Data Engineer |
| 2026-01-17 | - | RCA: Identified co-op/condo sqft issue as data source limitation | Data Engineer |
| 2026-02-11 | 0.4 | Implemented hierarchical sqft/year_built imputation with tracking flags | Data Engineer |
| 2026-02-11 | 0.5 | Updated ETL outputs after dropping invalid BBL (zero BBL) and rerunning full load | Data Engineer |

---

*Last Updated: 2026-02-11*
