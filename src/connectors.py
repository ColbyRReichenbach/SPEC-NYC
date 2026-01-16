"""
NYC Open Data Connectors

Downloads property sales data from NYC Open Data API (Socrata).
Implements rate limiting, retries, caching, and progress tracking.

Usage:
    python -m src.connectors

Dataset IDs:
    - Annualized Sales (preferred): w2pb-icbu
      Contains 2016-2024 data WITH lat/lon already included!
    - Rolling Sales (last 12 months only): usep-8jbt
    - PLUTO (not needed, Annualized has coordinates): 64uk-42ks
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# NYC Open Data API base URL
BASE_URL = "https://data.cityofnewyork.us/resource"

# Dataset IDs
DATASETS = {
    # Annualized Sales - BEST SOURCE
    # Contains 2016-2024 data with lat/lon, BBL, and census info
    "annualized_sales": "w2pb-icbu",
    
    # Rolling Sales - only last 12 months
    "rolling_sales": "usep-8jbt",
    
    # PLUTO - for coordinates (not needed if using annualized)
    "pluto": "64uk-42ks",
}

# Rate limiting settings (conservative to avoid any issues)
REQUEST_DELAY_SECONDS = 2.0  # Wait between requests
MAX_RETRIES = 3              # Retry failed requests
RETRY_BACKOFF = 2.0          # Exponential backoff multiplier
RECORDS_PER_REQUEST = 50000  # Socrata limit

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


# =============================================================================
# HTTP Client with Retry Logic
# =============================================================================

def create_session() -> requests.Session:
    """Create a requests session with retry logic and proper headers."""
    session = requests.Session()
    
    # Retry strategy with exponential backoff
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Be a good citizen - identify ourselves
    session.headers.update({
        "User-Agent": "SPEC-NYC-Valuation-Model/1.0 (Educational Project)",
        "Accept": "text/csv"
    })
    
    return session


def rate_limited_request(session: requests.Session, url: str) -> requests.Response:
    """Make a request with rate limiting and logging."""
    logger.debug(f"Requesting: {url[:100]}...")
    
    response = session.get(url, timeout=120)  # 2 min timeout for large requests
    response.raise_for_status()
    
    # Rate limiting: wait between requests
    time.sleep(REQUEST_DELAY_SECONDS)
    
    return response


# =============================================================================
# Annualized Sales Data (PREFERRED - includes coordinates!)
# =============================================================================

def download_annualized_sales(
    start_year: int = 2019,
    end_year: int = 2025,
    cache: bool = True
) -> pd.DataFrame:
    """
    Download NYC Annualized Sales data from Open Data API.
    
    This is the PREFERRED dataset because it:
    - Contains data from 2016-2024
    - Already includes latitude/longitude (no PLUTO join needed!)
    - Includes BBL, census tract, and community board
    
    Args:
        start_year: First year to include (default 2019)
        end_year: Last year to include (default 2025, but data goes to 2024)
        cache: If True, save to disk and reuse cached files
        
    Returns:
        DataFrame with all sales data including coordinates
    """
    cache_file = RAW_DIR / f"annualized_sales_{start_year}_{end_year}.csv"
    
    # Return cached data if available
    if cache and cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)
    
    logger.info(f"Downloading Annualized Sales data ({start_year}-{end_year})...")
    logger.info("This dataset includes lat/lon - no PLUTO join needed!")
    
    session = create_session()
    dataset_id = DATASETS["annualized_sales"]
    all_data = []
    
    for year in range(start_year, min(end_year + 1, 2025)):  # Data only goes to 2024
        logger.info(f"  Fetching {year}...")
        year_data = _fetch_year_data(session, dataset_id, year)
        
        if len(year_data) > 0:
            all_data.append(year_data)
            logger.info(f"    → {len(year_data):,} records")
        else:
            logger.warning(f"    → No data found for {year}")
    
    if not all_data:
        raise ValueError("No data downloaded. Check API connectivity.")
    
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total: {len(df):,} records downloaded")
    
    # Report coordinate coverage
    with_coords = df['latitude'].notna().sum()
    pct = (with_coords / len(df) * 100) if len(df) > 0 else 0
    logger.info(f"Records with coordinates: {with_coords:,}/{len(df):,} ({pct:.1f}%)")
    
    # Cache to disk
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        logger.info(f"Cached to {cache_file}")
    
    return df


def _fetch_year_data(
    session: requests.Session,
    dataset_id: str,
    year: int
) -> pd.DataFrame:
    """Fetch all sales for a given year with pagination."""
    all_chunks = []
    offset = 0
    
    while True:
        # Build query URL
        url = (
            f"{BASE_URL}/{dataset_id}.csv"
            f"?$limit={RECORDS_PER_REQUEST}"
            f"&$offset={offset}"
            f"&$where=sale_date >= '{year}-01-01' AND sale_date < '{year + 1}-01-01'"
            f"&$order=sale_date"
        )
        
        try:
            response = rate_limited_request(session, url)
            chunk = pd.read_csv(StringIO(response.text))
            
            if len(chunk) == 0:
                break
                
            all_chunks.append(chunk)
            offset += RECORDS_PER_REQUEST
            
            # If we got fewer than the limit, we're done
            if len(chunk) < RECORDS_PER_REQUEST:
                break
                
            logger.info(f"      Fetched {offset:,} records so far...")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            # If we have some data, return what we have
            if all_chunks:
                logger.warning("Returning partial data due to error")
                break
            raise
    
    if not all_chunks:
        return pd.DataFrame()
    
    return pd.concat(all_chunks, ignore_index=True)


# =============================================================================
# PLUTO Data (Only needed if using Rolling Sales)
# =============================================================================

def download_pluto_coordinates(cache: bool = True) -> pd.DataFrame:
    """
    Download PLUTO data with just BBL and coordinates.
    
    NOTE: Not needed if using download_annualized_sales() which includes coordinates.
    
    Args:
        cache: If True, save to disk and reuse cached files
        
    Returns:
        DataFrame with BBL and coordinates
    """
    cache_file = RAW_DIR / "pluto_coordinates.csv"
    
    if cache and cache_file.exists():
        logger.info(f"Loading cached PLUTO data from {cache_file}")
        return pd.read_csv(cache_file)
    
    logger.info("Downloading PLUTO coordinates...")
    
    session = create_session()
    dataset_id = DATASETS["pluto"]
    all_chunks = []
    offset = 0
    
    while True:
        url = (
            f"{BASE_URL}/{dataset_id}.csv"
            f"?$select=bbl,latitude,longitude"
            f"&$limit={RECORDS_PER_REQUEST}"
            f"&$offset={offset}"
            f"&$where=latitude IS NOT NULL"
        )
        
        try:
            response = rate_limited_request(session, url)
            chunk = pd.read_csv(StringIO(response.text))
            
            if len(chunk) == 0:
                break
                
            all_chunks.append(chunk)
            logger.info(f"  Fetched {offset + len(chunk):,} records...")
            
            offset += RECORDS_PER_REQUEST
            
            if len(chunk) < RECORDS_PER_REQUEST:
                break
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching PLUTO data: {e}")
            if all_chunks:
                logger.warning("Returning partial data due to error")
                break
            raise
    
    if not all_chunks:
        raise ValueError("No PLUTO data downloaded.")
    
    df = pd.concat(all_chunks, ignore_index=True)
    logger.info(f"Total: {len(df):,} PLUTO records")
    
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        logger.info(f"Cached to {cache_file}")
    
    return df


# =============================================================================
# Main Entry Point
# =============================================================================

def download_all(
    start_year: int = 2019,
    end_year: int = 2025,
    cache: bool = True
) -> pd.DataFrame:
    """
    Download all required data using the Annualized Sales dataset.
    
    This is the main entry point for data acquisition.
    Uses download_annualized_sales() which includes coordinates.
    
    Args:
        start_year: First year of sales data (2016 minimum available)
        end_year: Last year of sales data (2024 maximum available)
        cache: Whether to cache downloaded files
        
    Returns:
        Complete DataFrame ready for ETL processing
    """
    logger.info("=" * 60)
    logger.info("S.P.E.C. NYC Data Acquisition")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Download annualized sales (includes coordinates!)
    df = download_annualized_sales(start_year, end_year, cache)
    
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Data acquisition complete in {elapsed:.1f} seconds")
    logger.info(f"Final dataset: {len(df):,} records")
    logger.info("=" * 60)
    
    return df


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download NYC property sales data from Open Data API"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2019,
        help="First year to download (default: 2019, min: 2016)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to download (default: 2025, data available through 2024)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download even if cached data exists"
    )
    parser.add_argument(
        "--pluto-only",
        action="store_true",
        help="Only download PLUTO coordinates (not needed for main workflow)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.pluto_only:
            df = download_pluto_coordinates(cache=not args.no_cache)
        else:
            df = download_all(
                args.start_year,
                args.end_year,
                cache=not args.no_cache
            )
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"Total records: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nDate range: {df['sale_date'].min()} to {df['sale_date'].max()}")
        
        # Coordinate coverage
        if 'latitude' in df.columns:
            with_coords = df['latitude'].notna().sum()
            print(f"Records with coordinates: {with_coords:,} ({with_coords/len(df)*100:.1f}%)")
        
        print("\nSample data:")
        print(df.head(3).to_string())
        
    except KeyboardInterrupt:
        logger.info("\nDownload cancelled by user")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise
