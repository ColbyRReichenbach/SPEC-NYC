"""
S.P.E.C. NYC Configuration Settings
"""
from pathlib import Path
from dataclasses import dataclass

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# NYC GEOGRAPHY
# ============================================================================
# Manhattan center (Times Square area)
CITY_CENTER_LAT = 40.7580
CITY_CENTER_LON = -73.9855

# Borough codes (NYC DOF)
BOROUGH_CODES = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
}

# Initial scope (V1.0)
ACTIVE_BOROUGHS = [1, 3]  # Manhattan, Brooklyn

# ============================================================================
# H3 SPATIAL INDEXING
# ============================================================================
H3_RESOLUTION = 8  # ~460m hexagons

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
@dataclass
class ModelConfig:
    """XGBoost model configuration"""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42

MODEL_CONFIG = ModelConfig()

# Optuna tuning
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 1 hour max

# ============================================================================
# DATA QUALITY THRESHOLDS
# ============================================================================
MIN_SALE_PRICE = 10_000  # Filter $0 and micro-transactions
MAX_SALE_PRICE = 100_000_000  # Filter data errors
MIN_SQFT = 100
MAX_SQFT = 50_000

# Building classes to include (residential only)
RESIDENTIAL_BUILDING_CLASSES = [
    "A",  # One-family dwellings
    "B",  # Two-family dwellings
    "C",  # Walk-up apartments
    "D",  # Elevator apartments
    "R",  # Condominiums
]

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
NUMERIC_FEATURES = [
    "sqft",
    "year_built",
    "units_total",
    "distance_to_center_km",
    "h3_price_lag",
]

CATEGORICAL_FEATURES = [
    "borough",
    "building_class",
]

# ============================================================================
# UI CONFIGURATION (Dark mode theme)
# ============================================================================
THEME = {
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a",
    "bg_card": "#1a1a24",
    "text_primary": "#e8e8eb",
    "text_secondary": "#9ca3af",
    "accent": "#3b82f6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
}

# ============================================================================
# API KEYS (loaded from environment)
# ============================================================================
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://spec:spec_password@localhost:5432/spec_nyc")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
