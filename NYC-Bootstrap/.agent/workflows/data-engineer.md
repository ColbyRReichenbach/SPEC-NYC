---
description: Build and validate the NYC ETL pipeline for 1M+ transactions
---

# Data Engineer Workflow

**Role**: Build production-grade ETL pipeline for NYC property data with strict schema integrity.

---

## Prerequisites Check

// turbo
1. Verify Docker is running:
   ```bash
   docker info > /dev/null 2>&1 && echo "✓ Docker running" || echo "✗ Docker not running"
   ```

// turbo
2. Check project structure exists:
   ```bash
   ls -la /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/
   ```

---

## Task: Infrastructure Setup (Phase 1.2)

### Step 1: Start Docker Services
```bash
cd /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap
docker-compose up -d
```

### Step 2: Verify PostgreSQL Connection
```bash
docker-compose exec db psql -U spec_user -d spec_nyc -c "SELECT version();"
```

### Step 3: Create Data Directories
// turbo
```bash
mkdir -p data/raw data/processed logs models
```

### Success Criteria
- [ ] `docker-compose ps` shows both services running
- [ ] PostgreSQL responds to queries
- [ ] Directory structure complete

---

## Task: Database Schema (Phase 1.3)

### Step 1: Create Database Module
Create `src/database.py` with SQLAlchemy models:

```python
from sqlalchemy import create_engine, Column, String, Integer, Float, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import DATABASE_URL

Base = declarative_base()

class Sale(Base):
    __tablename__ = 'sales'
    id = Column(Integer, primary_key=True)
    bbl = Column(String(10), index=True, nullable=False)
    sale_price = Column(Float, nullable=False)
    sale_date = Column(Date)
    building_class = Column(String(2))
    address = Column(String(255))
    borough = Column(Integer)
    
class Property(Base):
    __tablename__ = 'properties'
    bbl = Column(String(10), primary_key=True)
    sqft = Column(Float)
    year_built = Column(Integer)
    units_total = Column(Integer)
    zoning = Column(String(20))
    lat = Column(Float)
    lon = Column(Float)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    bbl = Column(String(10), index=True)
    predicted_price = Column(Float)
    prediction_date = Column(DateTime)
    model_version = Column(String(20))
```

### Step 2: Initialize Database
```bash
python -c "from src.database import Base, engine; Base.metadata.create_all(engine)"
```

### Success Criteria
- [ ] All three tables created in PostgreSQL
- [ ] BBL columns are indexed
- [ ] Model imports without errors

---

## Task: Data Ingestion (Phase 1.4)

### Step 1: Download Raw Data

Download from NYC Open Data:
- Rolling Sales: https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page
- PLUTO: https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page

Save to `data/raw/`:
- `manhattan_sales.csv`
- `brooklyn_sales.csv`
- `pluto_manhattan.csv`
- `pluto_brooklyn.csv`

### Step 2: Create Connectors Module
Create `src/connectors.py`:

```python
import pandas as pd
from pathlib import Path
from config.settings import DATA_RAW_DIR

def load_rolling_sales(borough: str) -> pd.DataFrame:
    """Load NYC Rolling Sales data for a borough."""
    filepath = DATA_RAW_DIR / f"{borough.lower()}_sales.csv"
    df = pd.read_csv(filepath)
    df['bbl'] = df.apply(lambda r: f"{r['BOROUGH']}{r['BLOCK']:05d}{r['LOT']:04d}", axis=1)
    return df

def load_pluto(borough: str) -> pd.DataFrame:
    """Load PLUTO building characteristics."""
    filepath = DATA_RAW_DIR / f"pluto_{borough.lower()}.csv"
    df = pd.read_csv(filepath)
    df['bbl'] = df['BBL'].astype(str).str.zfill(10)
    return df

def join_sales_pluto(sales: pd.DataFrame, pluto: pd.DataFrame) -> pd.DataFrame:
    """Join sales with building characteristics on BBL."""
    return sales.merge(pluto, on='bbl', how='left')
```

### Step 3: Validate BBL Parsing
```python
# Test BBL format: Borough(1) + Block(5) + Lot(4) = 10 chars
assert len("1000010001") == 10  # Manhattan, Block 1, Lot 1
```

### Success Criteria
- [ ] Raw files exist in `data/raw/`
- [ ] BBL parsing produces valid 10-character strings
- [ ] Join produces ≥50,000 records

---

## Task: Data Cleaning (Phase 1.5)

### Technical Constraints
These rules are **mandatory** and **non-negotiable**:

| Rule | Filter | Reason |
|------|--------|--------|
| Non-Market Sales | `sale_price < 10,000` | Family transfers, nominal sales |
| Zero Sales | `sale_price == 0` | No transaction value |
| Building Class | Keep only A, B, C, D, R | Residential focus for V1.0 |
| Duplicates | Keep most recent per BBL | Avoid double-counting |
| Missing Sqft | Impute from class median | Required for modeling |

### Step 1: Create ETL Module
Create `src/etl.py`:

```python
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/etl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RESIDENTIAL_CLASSES = ['A', 'B', 'C', 'D', 'R']
MIN_SALE_PRICE = 10_000

def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules to sales data."""
    initial_count = len(df)
    
    # Filter zero sales
    df = df[df['sale_price'] > 0]
    logger.info(f"Removed {initial_count - len(df)} zero-price sales")
    
    # Filter non-market sales
    before = len(df)
    df = df[df['sale_price'] >= MIN_SALE_PRICE]
    logger.info(f"Removed {before - len(df)} non-market sales (<${MIN_SALE_PRICE:,})")
    
    # Filter to residential classes
    before = len(df)
    df = df[df['building_class'].str[0].isin(RESIDENTIAL_CLASSES)]
    logger.info(f"Removed {before - len(df)} non-residential properties")
    
    # Remove duplicates (keep most recent)
    before = len(df)
    df = df.sort_values('sale_date', ascending=False).drop_duplicates('bbl', keep='first')
    logger.info(f"Removed {before - len(df)} duplicate BBLs")
    
    logger.info(f"Final record count: {len(df)} ({len(df)/initial_count*100:.1f}% retained)")
    return df

def impute_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing sqft from building class median."""
    missing_before = df['sqft'].isna().sum()
    
    class_medians = df.groupby('building_class')['sqft'].median()
    df['sqft'] = df.apply(
        lambda r: class_medians.get(r['building_class'], df['sqft'].median()) 
        if pd.isna(r['sqft']) else r['sqft'],
        axis=1
    )
    
    logger.info(f"Imputed {missing_before} missing sqft values")
    return df
```

### Step 2: Run ETL Pipeline
```bash
python -m src.etl
```

### Step 3: Verify Cleaning Statistics
Check `logs/etl.log` for expected output:
```
Removed X zero-price sales
Removed X non-market sales (<$10,000)
Removed X non-residential properties
Removed X duplicate BBLs
Final record count: X (X% retained)
```

### Success Criteria
- [ ] Cleaning log shows all filters applied
- [ ] No records with `sale_price < 10,000`
- [ ] No records with building class outside A, B, C, D, R
- [ ] No duplicate BBLs
- [ ] Missing sqft values imputed

---

## Task: Feature Engineering (Phase 1.6)

### Step 1: Create Spatial Module
Create `src/spatial.py`:

```python
import h3
import numpy as np
from typing import Tuple

# Manhattan center (Columbus Circle)
NYC_CENTER: Tuple[float, float] = (40.7680, -73.9819)
H3_RESOLUTION = 8  # ~460m hexagons

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_distance_to_center(lat: float, lon: float) -> float:
    """Distance from property to Manhattan center."""
    return haversine_distance(lat, lon, NYC_CENTER[0], NYC_CENTER[1])

def get_h3_index(lat: float, lon: float) -> str:
    """Get H3 hex index for a location."""
    return h3.latlng_to_cell(lat, lon, H3_RESOLUTION)

def compute_h3_price_lag(df, h3_col: str = 'h3_index', price_col: str = 'sale_price'):
    """Compute median price of neighboring hexagons."""
    hex_medians = df.groupby(h3_col)[price_col].median().to_dict()
    
    def get_neighbor_median(h3_idx):
        neighbors = h3.grid_ring(h3_idx, 1)
        prices = [hex_medians.get(n) for n in neighbors if n in hex_medians]
        return np.median(prices) if prices else np.nan
    
    return df[h3_col].apply(get_neighbor_median)
```

### Step 2: Apply Feature Engineering
```python
df['distance_to_center_km'] = df.apply(
    lambda r: compute_distance_to_center(r['lat'], r['lon']), axis=1
)
df['h3_index'] = df.apply(
    lambda r: get_h3_index(r['lat'], r['lon']), axis=1
)
df['h3_price_lag'] = compute_h3_price_lag(df)
```

### Success Criteria
- [ ] All records have `distance_to_center_km`
- [ ] All records have valid `h3_index`
- [ ] `h3_price_lag` computed with <5% NaN

---

## Validation Checklist

Before marking Phase 1 of Data Engineering complete:

// turbo
```bash
# Count records in database
docker-compose exec db psql -U spec_user -d spec_nyc -c "SELECT COUNT(*) FROM sales;"
```

```bash
# Verify no bad data
docker-compose exec db psql -U spec_user -d spec_nyc -c "SELECT COUNT(*) FROM sales WHERE sale_price < 10000;"
# Should return 0
```

```bash
# Check feature coverage
docker-compose exec db psql -U spec_user -d spec_nyc -c "SELECT COUNT(*) FROM properties WHERE sqft IS NULL;"
# Should return 0
```

---

## When Done

1. **Check off completed items** in `docs/NYC_IMPLEMENTATION_PLAN.md`
2. **Update state** if changing phase: `.agent/workflows/state.yaml`
3. **Commit**: `git commit -am "Complete [task] - Phase X.X"`
4. **Handoff**: Route to `/ml-engineer` for model training
