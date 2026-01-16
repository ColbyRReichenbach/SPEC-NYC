# Data Sources Decision Log

This document records the data source decisions made for S.P.E.C. NYC, including the rationale behind each choice. This ensures transparency and helps future contributors understand why certain approaches were taken.

---

## Decision: Use Annualized Sales API Instead of Rolling Sales + PLUTO

**Date**: 2026-01-16  
**Status**: Implemented  
**Impact**: Simplified ETL, better data quality, reduced complexity

---

### Original Plan

The original implementation plan called for:

1. **NYC Rolling Sales** (`usep-8jbt`) - Download transaction prices
2. **PLUTO** (`64uk-42ks`) - Download building characteristics and coordinates  
3. **Join on BBL** - Merge the two datasets to get complete records

### Problem Discovered

When testing the Rolling Sales API:

```python
# Testing date range available
url = 'https://data.cityofnewyork.us/resource/usep-8jbt.csv?$select=min(sale_date),max(sale_date)'
# Result: "2024-11-01" to "2025-10-31"
```

**The Rolling Sales API only contains the last 12 months of data.**

For 6+ years of historical data (2019-2025), we would have needed to:
- Manually download ~35 Excel files from nyc.gov
- Parse inconsistent Excel formats across years
- Join with PLUTO separately for coordinates
- Handle format changes over time

### Better Alternative Found

While searching for solutions, we discovered the **NYC Citywide Annualized Calendar Sales Update** dataset (`w2pb-icbu`):

```python
# Testing Annualized Sales
url = 'https://data.cityofnewyork.us/resource/w2pb-icbu.csv?$select=min(sale_date),max(sale_date)'
# Result: "2016-01-01" to "2024-12-31"
```

**Key advantages:**

| Feature | Rolling Sales + PLUTO | Annualized Sales |
|---------|----------------------|------------------|
| Date Range | Last 12 months only | 2016 - 2024 (9 years) |
| Coordinates | Requires PLUTO join | ✅ Included (`latitude`, `longitude`) |
| BBL | Must construct from parts | ✅ Included as single field |
| API Access | Partial | ✅ Full dataset via API |
| Additional Fields | Limited | Census tract, community board, NTA |
| Records | ~80,000 (12 months) | ~500,000 (6 years) |

### Decision Made

**We switched to using the Annualized Sales dataset exclusively.**

### Benefits of This Decision

1. **Simpler Pipeline**: No need to join two datasets - coordinates are included
2. **API-Only Workflow**: No manual Excel downloads required
3. **More Data**: 498,666 records vs. ~80,000
4. **Better Coverage**: 97.4% have coordinates (vs. unknown PLUTO match rate)
5. **Additional Features**: Census tract and community board included for free
6. **Reproducibility**: Single API call, easily re-run

### Tradeoffs Accepted

1. **Maximum date**: Data goes to 2024-12-31 (not 2025)
   - *Mitigation*: Dataset updates quarterly; 2025 data will appear in next refresh
   
2. **Minimum date**: Only goes back to 2016
   - *Mitigation*: We only planned to use 2019-2024 anyway

### Implementation Details

**Dataset ID**: `w2pb-icbu`  
**API Endpoint**: `https://data.cityofnewyork.us/resource/w2pb-icbu.csv`  
**Connector**: `src/connectors.py` → `download_annualized_sales()`

**Columns Retrieved**:
```
borough, neighborhood, building_class_category, tax_class_as_of_final_roll,
block, lot, ease_ment, building_class_as_of_final, address, apartment_number,
zip_code, residential_units, commercial_units, total_units, land_square_feet,
gross_square_feet, year_built, tax_class_at_time_of_sale, building_class_at_time_of,
sale_price, sale_date, latitude, longitude, community_board, council_district,
bin, bbl, census_tract_2020, nta
```

### Data Downloaded

| Year | Records |
|------|---------|
| 2019 | 82,811 |
| 2020 | 68,684 |
| 2021 | 99,090 |
| 2022 | 93,427 |
| 2023 | 76,411 |
| 2024 | 78,243 |
| **Total** | **498,666** |

**Coordinate Coverage**: 485,863 records (97.4%)

**Cache Location**: `data/raw/annualized_sales_2019_2025.csv`

---

## PLUTO: Status and Future Use

PLUTO is **no longer required for V1.0** since Annualized Sales includes coordinates.

However, PLUTO may be useful in future versions for:
- Additional building characteristics not in sales data
- Zoning information
- Lot dimensions
- FAR (Floor Area Ratio)

The connector still supports PLUTO download if needed:
```python
from src.connectors import download_pluto_coordinates
pluto = download_pluto_coordinates()
```

---

## MTA Subway Stations: Still Required

The MTA Subway Stations dataset is still needed for transit proximity features.
- **Dataset**: MTA Subway Stations (data.ny.gov)
- **Use**: Compute `distance_to_nearest_subway` feature
- **Join Method**: Spatial (nearest neighbor based on lat/lon)

This will be implemented in Phase 1.6 (Feature Engineering).

---

## Summary

| Original Plan | Actual Implementation | Reason |
|--------------|----------------------|--------|
| Rolling Sales | ❌ Skipped | Only 12 months of data |
| PLUTO (for coords) | ❌ Not needed | Annualized has coords |
| Annualized Sales | ✅ Primary source | 6 years, includes coords |
| MTA Subway | ⏳ Phase 1.6 | Still needed for features |

---

## References

- Annualized Sales API: https://data.cityofnewyork.us/resource/w2pb-icbu
- Rolling Sales API: https://data.cityofnewyork.us/resource/usep-8jbt
- PLUTO API: https://data.cityofnewyork.us/resource/64uk-42ks
- DOF Sales Page: https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page
