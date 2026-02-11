"""
S.P.E.C. NYC - ETL Pipeline (v2)

Extracts data from raw CSVs (Annualized Sales),
Transforms it (Cleaning + Enrichment + Segmentation),
Loads it into PostgreSQL.

Key Features:
- Property identification (BBL + apartment)
- Sales history tracking (resales, appreciation)
- Property segmentation (for cascading models)
- Hierarchical imputation with transparency

Usage:
    python -m src.etl

See docs/DATA_QUALITY_LOG.md for transformation documentation.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.database import Sales, create_tables, get_session
from src.validation.data_contracts import DataContractResult, validate_data_contracts

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
REPORTS_DATA_DIR = Path("reports/data")
CURRENT_YEAR = 2025  # For building_age calculation

# NYC Center (Manhattan - Columbus Circle)
NYC_CENTER_LAT = 40.7831
NYC_CENTER_LON = -73.9712

# H3 Resolution (8 = ~460m edge length, good for neighborhood granularity)
H3_RESOLUTION = 8

# Building Class Categories to Keep (Residential)
RESIDENTIAL_PREFIXES = ["01", "02", "03", "07", "08", "09", "10", "12", "13", "14", "15", "16", "17"]

# Property Segment Mapping (building_class_category prefix -> segment)
SEGMENT_MAPPING = {
    "01": "SINGLE_FAMILY",   # One Family Dwellings
    "02": "SINGLE_FAMILY",   # Two Family Dwellings
    "03": "SINGLE_FAMILY",   # Three Family Dwellings
    "07": "WALKUP",          # Rentals - Walkup Apartments
    "08": "ELEVATOR",        # Rentals - Elevator Apartments
    "09": "WALKUP",          # Coops - Walkup Apartments
    "10": "ELEVATOR",        # Coops - Elevator Apartments
    "12": "WALKUP",          # Condos - Walkup Apartments
    "13": "ELEVATOR",        # Condos - Elevator Apartments
    "14": "SMALL_MULTI",     # Rentals - 4-10 Unit
    "15": "SMALL_MULTI",     # Condos - 2-10 Unit Residential
    "16": "SMALL_MULTI",     # Condos - 2-10 Unit with Commercial
    "17": "SMALL_MULTI",     # Condo Coops
}

RAW_CONTRACT_REQUIRED_COLUMNS = [
    "sale_date",
    "sale_price",
    "bbl",
    "latitude",
    "longitude",
    "borough",
    "block",
    "lot",
    "building_class_category",
]

POST_ETL_NULL_THRESHOLDS = {
    "sale_date": 0.0,
    "sale_price": 0.0,
    "bbl": 0.0,
    "latitude": 0.0,
    "longitude": 0.0,
    "property_id": 0.0,
    "property_segment": 0.0,
    "price_tier": 0.0,
    "gross_square_feet": 0.0,
    "year_built": 0.0,
}


# =============================================================================
# 1. EXTRACTION
# =============================================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw Annualized Sales data from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run src/connectors.py first.")
    
    logger.info(f"Loading raw data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df):,} records")
    
    return df


# =============================================================================
# 2. CLEANING
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data:
    - Standardize column names
    - Filter invalid/non-market sales
    - Filter non-residential properties
    - Filter missing critical fields
    """
    initial_count = len(df)
    
    # 1. Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    rename_map = {
        'building_class_at_time_of': 'building_class',
        'tax_class_at_time_of_sale': 'tax_class',
        'bin': 'bin_number',
        'census_tract_2020': 'census_tract'
    }
    df = df.rename(columns=rename_map)
    
    # 2. Clean numeric columns
    numeric_cols = ['land_square_feet', 'gross_square_feet', 'sale_price']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Filter: Sale price >= $10,000 (non-market sales)
    df = df[df['sale_price'].notna()]
    df = df[df['sale_price'] >= 10_000]
    logger.info(f"Filtered low/zero sales: {initial_count:,} -> {len(df):,} records")
    
    # 4. Filter: Residential only
    if 'building_class_category' in df.columns:
        df['category_code'] = df['building_class_category'].astype(str).str[:2]
        df = df[df['category_code'].isin(RESIDENTIAL_PREFIXES)]
        logger.info(f"Filtered non-residential: -> {len(df):,} records")
        df = df.drop(columns=['category_code'])
    
    # 5. Filter: Must have coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    logger.info(f"Filtered missing coordinates: -> {len(df):,} records")
    
    # 6. Filter: Must have valid BBL
    df = df.dropna(subset=['bbl'])
    df['bbl'] = pd.to_numeric(df['bbl'], errors='coerce')
    df = df[df['bbl'].notna()]
    df['bbl'] = df['bbl'].astype('int64')
    df = df[df['bbl'] > 0]
    logger.info(f"Filtered missing/invalid BBL: -> {len(df):,} records")
    
    # 7. Parse dates
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    df['sale_quarter'] = df['sale_date'].dt.quarter
    
    # 8. year_built to integer (0 for missing)
    df['year_built'] = df['year_built'].fillna(0).astype('int64')
    
    return df


# =============================================================================
# 3. PROPERTY IDENTIFICATION
# =============================================================================

def create_property_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unique property identifier:
    - For single-family: just BBL
    - For condos/co-ops: BBL + apartment_number
    
    This allows tracking resales of the SAME UNIT vs different units in same building.
    """
    logger.info("Creating property_id (BBL + apartment where applicable)...")
    
    def make_property_id(row):
        bbl = str(int(row['bbl']))
        apt = str(row['apartment_number']).strip() if pd.notna(row['apartment_number']) else ''
        if apt and apt.lower() not in ['nan', '', 'none']:
            return f"{bbl}_{apt}"
        return bbl
    
    df['property_id'] = df.apply(make_property_id, axis=1)
    
    unique_properties = df['property_id'].nunique()
    logger.info(f"Created {unique_properties:,} unique property IDs from {len(df):,} sales")
    
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove TRUE duplicates only:
    - Same property_id + same sale_date + same sale_price = duplicate
    
    Keep all legitimate resales (same property, different transactions).
    """
    before = len(df)
    
    # Sort by property_id, sale_date, then keep first occurrence of duplicates
    df = df.sort_values(['property_id', 'sale_date', 'sale_price'])
    df = df.drop_duplicates(subset=['property_id', 'sale_date', 'sale_price'], keep='first')
    
    after = len(df)
    logger.info(f"Removed TRUE duplicates: {before:,} -> {after:,} records ({before - after:,} removed)")
    
    return df


# =============================================================================
# 4. SALES HISTORY ENRICHMENT
# =============================================================================

def enrich_sales_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each property, track sales history:
    - sale_sequence: 1st, 2nd, 3rd sale
    - is_latest_sale: True for most recent
    - previous_sale_price/date: Prior sale info
    - price_change_pct: Appreciation
    """
    logger.info("Enriching sales history...")
    
    # Sort by property and date
    df = df.sort_values(['property_id', 'sale_date'])
    
    # Sale sequence within each property
    df['sale_sequence'] = df.groupby('property_id').cumcount() + 1
    
    # Is latest sale?
    df['is_latest_sale'] = df.groupby('property_id')['sale_date'].transform('max') == df['sale_date']
    
    # Previous sale info (shift within group)
    df['previous_sale_price'] = df.groupby('property_id')['sale_price'].shift(1)
    df['previous_sale_date'] = df.groupby('property_id')['sale_date'].shift(1)
    
    # Price change percentage
    mask_has_previous = df['previous_sale_price'].notna()
    df.loc[mask_has_previous, 'price_change_pct'] = (
        (df.loc[mask_has_previous, 'sale_price'] - df.loc[mask_has_previous, 'previous_sale_price']) 
        / df.loc[mask_has_previous, 'previous_sale_price'] * 100
    )
    
    # Days since last sale
    df.loc[mask_has_previous, 'days_since_last_sale'] = (
        (df.loc[mask_has_previous, 'sale_date'] - df.loc[mask_has_previous, 'previous_sale_date']).dt.days
    )
    
    # Stats
    resales = (df['sale_sequence'] > 1).sum()
    latest_sales = df['is_latest_sale'].sum()
    logger.info(f"  ├─ Properties with resales: {resales:,}")
    logger.info(f"  └─ Latest sales (for map): {latest_sales:,}")
    
    return df


# =============================================================================
# 5. PROPERTY SEGMENTATION
# =============================================================================

def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign property segments for cascading model architecture:
    - SINGLE_FAMILY: Houses (01, 02, 03)
    - WALKUP: No-elevator apartments (07, 09, 12)
    - ELEVATOR: Doorman buildings (08, 10, 13)
    - SMALL_MULTI: 2-10 unit buildings (14, 15, 16, 17)
    """
    logger.info("Assigning property segments...")
    
    # Extract category prefix
    df['category_prefix'] = df['building_class_category'].astype(str).str[:2]
    
    # Map to segment
    df['property_segment'] = df['category_prefix'].map(SEGMENT_MAPPING)
    df['property_segment'] = df['property_segment'].fillna('OTHER')
    
    # Drop helper column
    df = df.drop(columns=['category_prefix'])
    
    # Segment distribution
    segment_counts = df['property_segment'].value_counts()
    logger.info("Segment distribution:")
    for seg, count in segment_counts.items():
        logger.info(f"  {seg}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


def assign_price_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign within-segment price tiers:
    - entry: Bottom 25%
    - core: 25-50%
    - premium: 50-75%
    - luxury: Top 25%
    
    Tiers are computed WITHIN each segment for fair comparison.
    """
    logger.info("Assigning price tiers within segments...")
    
    tier_labels = ['entry', 'core', 'premium', 'luxury']

    def _segment_tiers(x: pd.Series) -> pd.Series:
        ranked = x.rank(method='first')
        try:
            return pd.qcut(ranked, q=4, labels=tier_labels, duplicates='drop')
        except ValueError:
            # Fallback for tiny segments (e.g., deterministic dry-run with --limit).
            pct = (ranked - 1) / max(len(ranked) - 1, 1)
            return pd.cut(
                pct,
                bins=[-0.001, 0.25, 0.50, 0.75, 1.00],
                labels=tier_labels,
                include_lowest=True,
            )

    # Compute quartiles within each segment
    df['price_tier'] = df.groupby('property_segment')['sale_price'].transform(_segment_tiers)
    
    # Show tier distribution
    tier_counts = df['price_tier'].value_counts()
    logger.info("Price tier distribution:")
    for tier in ['entry', 'core', 'premium', 'luxury']:
        if tier in tier_counts.index:
            logger.info(f"  {tier}: {tier_counts[tier]:,}")
    
    return df


# =============================================================================
# 6. IMPUTATION
# =============================================================================

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using hierarchical median imputation.
    
    Strategy documented in docs/DATA_QUALITY_LOG.md:
    - Uses median from most granular group available
    - Falls back to broader groups if no median exists
    - Tracks imputation source for transparency
    """
    
    # ==========================================================================
    # GROSS SQUARE FEET
    # ==========================================================================
    df['sqft_imputed'] = False
    df['sqft_imputation_level'] = None
    
    mask_missing = (df['gross_square_feet'].isna()) | (df['gross_square_feet'] == 0)
    missing_count = mask_missing.sum()
    
    if missing_count > 0:
        logger.info(f"Imputing {missing_count:,} records with missing/zero SQFT...")
        
        valid_df = df[~mask_missing]
        
        # Compute medians at each level
        level1_medians = valid_df.groupby(['neighborhood', 'building_class'])['gross_square_feet'].median()
        level2_medians = valid_df.groupby(['borough', 'building_class'])['gross_square_feet'].median()
        level3_medians = valid_df.groupby('building_class')['gross_square_feet'].median()
        level4_median = valid_df['gross_square_feet'].median()
        
        imputed_count = {'level1': 0, 'level2': 0, 'level3': 0, 'level4': 0}
        
        for idx in df[mask_missing].index:
            neighborhood = df.loc[idx, 'neighborhood']
            building_class = df.loc[idx, 'building_class']
            borough = df.loc[idx, 'borough']
            
            imputed_value = None
            imputation_level = None
            
            # Level 1: Neighborhood + Building Class
            try:
                if (neighborhood, building_class) in level1_medians.index:
                    val = level1_medians[(neighborhood, building_class)]
                    if pd.notna(val) and val > 0:
                        imputed_value = val
                        imputation_level = 'neighborhood_class'
                        imputed_count['level1'] += 1
            except (KeyError, TypeError):
                pass
            
            # Level 2: Borough + Building Class
            if imputed_value is None:
                try:
                    if (borough, building_class) in level2_medians.index:
                        val = level2_medians[(borough, building_class)]
                        if pd.notna(val) and val > 0:
                            imputed_value = val
                            imputation_level = 'borough_class'
                            imputed_count['level2'] += 1
                except (KeyError, TypeError):
                    pass
            
            # Level 3: Building Class only
            if imputed_value is None:
                try:
                    if building_class in level3_medians.index:
                        val = level3_medians[building_class]
                        if pd.notna(val) and val > 0:
                            imputed_value = val
                            imputation_level = 'class_only'
                            imputed_count['level3'] += 1
                except (KeyError, TypeError):
                    pass
            
            # Level 4: Citywide fallback
            if imputed_value is None:
                imputed_value = level4_median if pd.notna(level4_median) else 1200  # Default 1200 sqft
                imputation_level = 'citywide'
                imputed_count['level4'] += 1
            
            df.loc[idx, 'gross_square_feet'] = imputed_value
            df.loc[idx, 'sqft_imputed'] = True
            df.loc[idx, 'sqft_imputation_level'] = imputation_level
        
        logger.info(f"  ├─ Neighborhood+Class: {imputed_count['level1']:,}")
        logger.info(f"  ├─ Borough+Class: {imputed_count['level2']:,}")
        logger.info(f"  ├─ Class-only: {imputed_count['level3']:,}")
        logger.info(f"  └─ Citywide: {imputed_count['level4']:,}")
    
    # ==========================================================================
    # YEAR BUILT
    # ==========================================================================
    df['year_built_imputed'] = False
    mask_missing_year = (df['year_built'] == 0) | (df['year_built'].isna())
    missing_year_count = mask_missing_year.sum()
    
    if missing_year_count > 0:
        logger.info(f"Imputing {missing_year_count:,} records with missing year_built...")
        
        valid_years = df[~mask_missing_year]
        year_medians_neighborhood = valid_years.groupby('neighborhood')['year_built'].median()
        year_medians_borough = valid_years.groupby('borough')['year_built'].median()
        citywide_year_median = valid_years['year_built'].median()
        
        for idx in df[mask_missing_year].index:
            neighborhood = df.loc[idx, 'neighborhood']
            borough = df.loc[idx, 'borough']
            
            imputed_year = None
            
            if neighborhood in year_medians_neighborhood.index:
                imputed_year = year_medians_neighborhood[neighborhood]
            
            if imputed_year is None or pd.isna(imputed_year) or imputed_year == 0:
                if borough in year_medians_borough.index:
                    imputed_year = year_medians_borough[borough]
            
            if imputed_year is None or pd.isna(imputed_year) or imputed_year == 0:
                imputed_year = citywide_year_median if pd.notna(citywide_year_median) else 1960
            
            df.loc[idx, 'year_built'] = int(imputed_year)
            df.loc[idx, 'year_built_imputed'] = True
        
        logger.info(f"  └─ Imputed {missing_year_count:,} year_built values")
    
    df['year_built'] = df['year_built'].astype('int64')
    
    return df


# =============================================================================
# 7. FEATURE ENGINEERING
# =============================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived and spatial features."""
    
    # 1. H3 Spatial Index
    logger.info(f"Generating H3 spatial indexes (res={H3_RESOLUTION})...")
    df['h3_index'] = df.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], H3_RESOLUTION),
        axis=1
    )
    
    # 2. Distance to Center
    lat_diff = (df['latitude'] - NYC_CENTER_LAT) * 111
    lon_diff = (df['longitude'] - NYC_CENTER_LON) * 111 * np.cos(np.radians(NYC_CENTER_LAT))
    df['distance_to_center_km'] = np.sqrt(lat_diff**2 + lon_diff**2)
    
    # 3. Building Age
    df['building_age'] = CURRENT_YEAR - df['year_built']
    df.loc[df['building_age'] < 0, 'building_age'] = 0  # Handle future dates
    df.loc[df['building_age'] > 300, 'building_age'] = 100  # Cap at 100 years for 0 year_built
    
    # 4. Price per Square Foot
    df['price_per_sqft'] = np.where(
        df['gross_square_feet'] > 0,
        df['sale_price'] / df['gross_square_feet'],
        np.nan
    )
    
    return df


# =============================================================================
# 8. LOADING
# =============================================================================

def load_to_postgres(df: pd.DataFrame, *, replace_existing: bool = False):
    """Insert processed data into PostgreSQL."""
    logger.info("Loading data to PostgreSQL...")
    
    records = df.to_dict(orient='records')
    session = get_session()
    
    valid_columns = {c.key for c in Sales.__table__.columns}
    
    try:
        if replace_existing:
            logger.info("replace_existing=True: truncating 'sales' table before load")
            session.execute(text("TRUNCATE TABLE sales RESTART IDENTITY"))
            session.commit()

        count = 0
        batch_size = 5000
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            objects = []
            for row in batch:
                # Filter to valid columns only
                row_cleaned = {
                    k: (v if pd.notna(v) else None)
                    for k, v in row.items() 
                    if k in valid_columns
                }
                objects.append(Sales(**row_cleaned))
            
            session.bulk_save_objects(objects)
            count += len(batch)
            if count % 20000 == 0:
                logger.info(f"Inserted {count:,} records...")
        
        session.commit()
        logger.info(f"✅ Successfully loaded {count:,} records into 'sales' table.")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error loading data: {e}")
        raise
    finally:
        session.close()


# =============================================================================
# 9. REPORTING
# =============================================================================

def _record_stage(stage_stats: list[dict], stage_name: str, df: pd.DataFrame) -> None:
    """Capture row and quality counters for ETL stage reporting."""
    stage_stats.append(
        {
            "stage": stage_name,
            "rows": len(df),
            "unique_properties": int(df["property_id"].nunique()) if "property_id" in df.columns else None,
            "latest_sale_date": str(pd.to_datetime(df["sale_date"], errors="coerce").max().date())
            if "sale_date" in df.columns and len(df) > 0
            else None,
        }
    )


def _write_etl_report(
    run_started_at: datetime,
    input_path: Path,
    dry_run: bool,
    stage_stats: list[dict],
    contract_results: list[tuple[str, DataContractResult]],
) -> tuple[Path, Path]:
    """
    Write ETL run artifacts:
    - reports/data/etl_run_YYYYMMDD.md
    - reports/data/etl_run_YYYYMMDD.csv
    """
    REPORTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    run_stamp = run_started_at.strftime("%Y%m%d")
    markdown_path = REPORTS_DATA_DIR / f"etl_run_{run_stamp}.md"
    csv_path = REPORTS_DATA_DIR / f"etl_run_{run_stamp}.csv"

    stage_df = pd.DataFrame(stage_stats)
    stage_df.to_csv(csv_path, index=False)

    contract_lines = []
    for label, result in contract_results:
        contract_lines.append(f"- **{label}**: {'PASS' if result.passed else 'FAIL'}")
        if result.violations:
            for violation in result.violations:
                contract_lines.append(
                    f"  - [{violation.check}] {violation.message} (failed_rows={violation.failed_rows})"
                )

    stage_table = stage_df.to_string(index=False) if not stage_df.empty else "No stage stats captured."

    md = [
        f"# ETL Run Report - {run_stamp}",
        "",
        "## Run Metadata",
        f"- Started (UTC): {run_started_at.isoformat()}",
        f"- Input: `{input_path}`",
        f"- Dry run: `{dry_run}`",
        "",
        "## Stage Summary",
        "```text",
        stage_table,
        "```",
        "",
        "## Data Contract Results",
    ]
    if contract_lines:
        md.extend(contract_lines)
    else:
        md.append("_No contract results captured._")

    markdown_path.write_text("\n".join(md), encoding="utf-8")
    return markdown_path, csv_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_etl(
    input_path: Path = RAW_DATA_PATH,
    limit: int | None = None,
    dry_run: bool = False,
    write_report: bool = False,
    replace_sales: bool = False,
) -> pd.DataFrame:
    """Execute the full ETL pipeline."""
    logger.info("=" * 60)
    logger.info("S.P.E.C. NYC ETL Pipeline v2")
    logger.info("=" * 60)

    run_started_at = datetime.utcnow()
    stage_stats: list[dict] = []
    contract_results: list[tuple[str, DataContractResult]] = []

    try:
        # 1. Extract
        df = load_raw_data(input_path)
        if limit is not None:
            df = df.head(limit).copy()
            logger.info(f"Applied deterministic row limit: {limit:,}")
        _record_stage(stage_stats, "extract_raw", df)

        # 2. Clean
        df = clean_data(df)
        _record_stage(stage_stats, "cleaned", df)

        clean_contract = validate_data_contracts(
            df,
            required_columns=RAW_CONTRACT_REQUIRED_COLUMNS,
            max_freshness_days=730,
            raise_on_error=True,
        )
        contract_results.append(("post-clean", clean_contract))
        logger.info("Data contracts (post-clean): PASS")

        # 3. Property Identification
        df = create_property_id(df)

        # 4. Deduplicate (TRUE duplicates only)
        df = deduplicate(df)

        # 5. Sales History
        df = enrich_sales_history(df)

        # 6. Segmentation
        df = assign_segments(df)
        df = assign_price_tiers(df)

        # 7. Imputation
        df = impute_missing_values(df)

        # 8. Feature Engineering
        df = feature_engineering(df)
        _record_stage(stage_stats, "feature_engineered", df)

        final_contract = validate_data_contracts(
            df,
            null_thresholds=POST_ETL_NULL_THRESHOLDS,
            max_freshness_days=730,
            raise_on_error=True,
        )
        contract_results.append(("post-feature-engineering", final_contract))
        logger.info("Data contracts (post-feature-engineering): PASS")

        # 9. Load
        if dry_run:
            logger.info("Dry run enabled: skipping PostgreSQL load.")
            _record_stage(stage_stats, "load_skipped_dry_run", df)
        else:
            create_tables()
            load_to_postgres(df, replace_existing=replace_sales)
            _record_stage(stage_stats, "loaded_postgres", df)

        # Summary
        logger.info("=" * 60)
        logger.info("ETL Pipeline Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"Final record count: {len(df):,}")
        logger.info(f"Unique properties: {df['property_id'].nunique():,}")
        logger.info(f"Properties with history: {(df['sale_sequence'] > 1).sum():,}")

        if write_report:
            markdown_path, csv_path = _write_etl_report(
                run_started_at=run_started_at,
                input_path=input_path,
                dry_run=dry_run,
                stage_stats=stage_stats,
                contract_results=contract_results,
            )
            logger.info(f"Wrote ETL report: {markdown_path}")
            logger.info(f"Wrote ETL CSV summary: {csv_path}")

        return df

    except Exception as e:
        logger.error(f"ETL Failed: {e}")
        if write_report and stage_stats:
            try:
                markdown_path, csv_path = _write_etl_report(
                    run_started_at=run_started_at,
                    input_path=input_path,
                    dry_run=dry_run,
                    stage_stats=stage_stats,
                    contract_results=contract_results,
                )
                logger.info(f"Partial ETL report written: {markdown_path}")
                logger.info(f"Partial ETL CSV written: {csv_path}")
            except Exception as report_error:
                logger.error(f"Failed to write partial ETL report: {report_error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S.P.E.C. NYC ETL pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help=f"Input CSV path (default: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic row limit (applied before transformation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full transformation and contracts without loading to PostgreSQL",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write reports/data/etl_run_YYYYMMDD.md and CSV stage summary",
    )
    parser.add_argument(
        "--replace-sales",
        action="store_true",
        help="Truncate the sales table before load to ensure idempotent full refreshes",
    )
    args = parser.parse_args()

    run_etl(
        input_path=args.input,
        limit=args.limit,
        dry_run=args.dry_run,
        write_report=args.write_report,
        replace_sales=args.replace_sales,
    )
