"""
S.P.E.C. NYC - ETL Pipeline

Extracts data from raw CSVs (Annualized Sales),
Transforms it (Cleaning + Feature Engineering),
Loads it into PostgreSQL.

Usage:
    python -m src.etl
"""

import logging
import pandas as pd
import numpy as np
import h3
from sqlalchemy.orm import Session
from pathlib import Path

from src.database import get_session, engine, Sales, create_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

RAW_DATA_PATH = Path("data/raw/annualized_sales_2019_2025.csv")

# NYC Center (Manhattan)
NYC_CENTER_LAT = 40.7831
NYC_CENTER_LON = -73.9712

# H3 Resolution
H3_RESOLUTION = 8  # ~460m edge length

# Building Class Categories to Keep (Residential)
RESIDENTIAL_PREFIXES = ["01", "02", "03", "07", "08", "09", "10", "12", "13", "14", "15", "16", "17"]


# =============================================================================
# 1. Extraction
# =============================================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw Annualized Sales data from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run src/connectors.py first.")
    
    logger.info(f"Loading raw data from {path}...")
    
    # Load with low_memory=False to avoid dtypes warnings
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df):,} records")
    
    return df


# =============================================================================
# 2. Transformation
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data:
    - Filter invalid prices
    - Keep only residential
    - Remove missing critical fields
    - Fix data types
    """
    initial_count = len(df)
    
    # 1. Standardize column names (strip whitespace)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # 1b. Rename columns to match Schema
    rename_map = {
        'building_class_at_time_of': 'building_class',
        'tax_class_at_time_of_sale': 'tax_class',
        'bin': 'bin_number',
        'census_tract_2020': 'census_tract'
    }
    df = df.rename(columns=rename_map)
    
    # 2. Filter Sales Price (Target)
    # Remove transfer/low sales (< $10k)
    df = df[df['sale_price'].notna()]
    df = df[df['sale_price'] >= 10_000]
    logger.info(f"Filtered low/zero sales: {initial_count} -> {len(df)} records")
    
    # 3. Filter Residential Only
    # Filter by building_class_category prefix
    # Categories start with a number, e.g., "01 ONE FAMILY DWELLINGS"
    if 'building_class_category' in df.columns:
        # Extract first 2 digits
        df['category_code'] = df['building_class_category'].astype(str).str[:2]
        df = df[df['category_code'].isin(RESIDENTIAL_PREFIXES)]
        logger.info(f"Filtered non-residential: -> {len(df)} records")
        
        # Drop helper col
        df = df.drop(columns=['category_code'])
    
    # 4. Filter Missing Coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    logger.info(f"Filtered missing coordinates: -> {len(df)} records")
    
    # 4b. Filter Missing BBL (required for database)
    df = df.dropna(subset=['bbl'])
    df['bbl'] = df['bbl'].astype('int64')  # Convert to integer for database
    logger.info(f"Filtered missing BBL: -> {len(df)} records")
    
    # 5. Clean Numeric Columns (strip commas from strings)
    numeric_cols = ['land_square_feet', 'gross_square_feet']
    for col in numeric_cols:
        if col in df.columns:
            # Check if string type before string ops, otherwise just ensure numeric
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors='coerce')
            else:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

    # 6. Date Parsing
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    
    # 6b. Convert year_built to integer (handle NaN as 0)
    df['year_built'] = df['year_built'].fillna(0).astype('int64')
    
    # 7. Deduplicate (Keep most recent sale per BBL + Price combo to avoid duplicates)
    # Actually, same BBL can sell multiple times. We keep all *valid* sales.
    # But exact duplicates (same BBL, date, price) should be removed.
    df = df.drop_duplicates(subset=['bbl', 'sale_date', 'sale_price'])
    logger.info(f"Removed duplicates: -> {len(df)} records")
    
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing sqft using neighborhood + building class median."""
    
    # 1. Gross Square Feet
    # If sqft is 0 or NaN, it needs imputation
    mask_missing = (df['gross_square_feet'].isna()) | (df['gross_square_feet'] == 0)
    missing_count = mask_missing.sum()
    
    if missing_count > 0:
        logger.info(f"Imputing {missing_count} records with missing SQFT...")
        
        # Calculate medians by Neighborhood + Building Class prefix (first 1 char)
        df['class_prefix'] = df['building_class'].astype(str).str[:1]
        
        medians = df[~mask_missing].groupby(['neighborhood', 'class_prefix'])['gross_square_feet'].transform('median')
        
        # Fill missing
        df.loc[mask_missing, 'gross_square_feet'] = medians[mask_missing]
        
        # If still missing (no median in that group), use citywide median for that class prefix
        mask_still_missing = (df['gross_square_feet'].isna()) | (df['gross_square_feet'] == 0)
        if mask_still_missing.sum() > 0:
            city_medians = df[~mask_still_missing].groupby('class_prefix')['gross_square_feet'].transform('median')
            df.loc[mask_still_missing, 'gross_square_feet'] = city_medians[mask_still_missing]
            
        # Drop helper
        df = df.drop(columns=['class_prefix'])
        
        logger.info(f"Remaining missing SQFT: {((df['gross_square_feet'].isna()) | (df['gross_square_feet'] == 0)).sum()}")
    
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add spatial and temporal features."""
    
    # 1. H3 Indexing (Spatial Hexagons)
    logger.info(f"Generating H3 spatial indexes (res={H3_RESOLUTION})...")
    df['h3_index'] = df.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], H3_RESOLUTION),
        axis=1
    )
    
    # 2. Distance to Center (Manhattan) - Haversine Approximation
    # Simple Euclidean on lat/lon is "good enough" for local sort, but let's do rough km
    # 1 deg lat ~ 111 km
    lat_diff = (df['latitude'] - NYC_CENTER_LAT) * 111
    lon_diff = (df['longitude'] - NYC_CENTER_LON) * 111 * np.cos(np.radians(NYC_CENTER_LAT))
    
    df['distance_to_center_km'] = np.sqrt(lat_diff**2 + lon_diff**2)
    
    return df


# =============================================================================
# 3. Loading
# =============================================================================

def load_to_postgres(df: pd.DataFrame):
    """Insert processed data into PostgreSQL."""
    logger.info("Loading data to PostgreSQL...")
    
    # Map DataFrame columns to SQLAlchemy model columns
    # Any extra columns in DF (like 'zip_code') match exactly?
    # We need to ensure types match.
    
    records = df.to_dict(orient='records')
    
    # Create session
    session = get_session()
    
    # Get valid column names from Sales model
    valid_columns = {c.key for c in Sales.__table__.columns}
    
    try:
        # Truncate table first? (Optional, usually good for full reload)
        # session.query(Sales).delete()
        # session.commit()
        
        # But for safety, let's just insert. Using bulk_insert_mappings for speed.
        count = 0
        batch_size = 5000
        
        # SQLAlchemy Core / Bulk insert is faster
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            # Map batch to Sales objects
            objects = []
            for row in batch:
                # Handle potentially missing optional fields that confuse SQL
                # AND filter out columns not in the model
                row_cleaned = {
                    k: v for k, v in row.items() 
                    if k in valid_columns and not pd.isna(v)
                }
                objects.append(Sales(**row_cleaned))
            
            session.bulk_save_objects(objects)
            count += len(batch)
            if count % 20000 == 0:
                logger.info(f"Inserted {count} records...")
        
        session.commit()
        logger.info(f"✅ Successfully loaded {count} records into 'sales' table.")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error loading data: {e}")
        raise
    finally:
        session.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def run_etl():
    """Execute the full ETL pipeline."""
    logger.info("=" * 60)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 60)
    
    try:
        # 1. Load
        df = load_raw_data(RAW_DATA_PATH)
        
        # 2. Clean
        df = clean_data(df)
        
        # 3. Impute
        df = impute_missing_values(df)
        
        # 4. Enrich
        df = feature_engineering(df)
        
        # 5. Load
        # First ensure tables exist
        create_tables()
        load_to_postgres(df)
        
        logger.info("=" * 60)
        logger.info("ETL Pipeline Completed Successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"ETL Failed: {e}")
        raise

if __name__ == "__main__":
    run_etl()
