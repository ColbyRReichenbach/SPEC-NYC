"""
S.P.E.C. NYC Database Models

SQLAlchemy models for the PostgreSQL database.
Schema designed based on actual Annualized Sales API data structure.

Usage:
    from src.database import engine, SessionLocal, Sales, Prediction
    
    # Create tables
    from src.database import create_tables
    create_tables()
"""

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    Float,
    String,
    DateTime,
    Date,
    Text,
    Index,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# =============================================================================
# Database Configuration
# =============================================================================

# Load from environment or use defaults
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://spec:spec_password@localhost:5433/spec_nyc"  # Port 5433 to avoid local postgres conflict
)

# Create engine
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# =============================================================================
# Sales Table
# =============================================================================

class Sales(Base):
    """
    NYC Property Sales - cleaned and geocoded.
    
    Source: Annualized Sales API (w2pb-icbu)
    This is the primary table for model training.
    """
    __tablename__ = "sales"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Borough-Block-Lot (unique property identifier)
    bbl = Column(BigInteger, nullable=False, index=True)  # 10-digit BBL
    borough = Column(Integer, nullable=False)  # 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island
    block = Column(Integer, nullable=False)
    lot = Column(Integer, nullable=False)
    
    # Location
    address = Column(String(255))
    apartment_number = Column(String(50))
    neighborhood = Column(String(100), index=True)
    zip_code = Column(Integer)
    
    # Coordinates (from Annualized Sales API - no PLUTO join needed!)
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Building Characteristics
    building_class_category = Column(String(100))  # e.g., "13 CONDOS - ELEVATOR APARTMENTS"
    building_class = Column(String(10), index=True)  # e.g., "R4"
    
    # Size metrics
    residential_units = Column(Float)
    commercial_units = Column(Float)
    total_units = Column(Float)
    land_square_feet = Column(Float)
    gross_square_feet = Column(Float)  # KEY FEATURE for model
    year_built = Column(Integer)  # KEY FEATURE for model
    
    # Tax info
    tax_class = Column(String(10))  # Tax class at time of sale
    
    # SALE INFO (TARGET VARIABLE)
    sale_price = Column(BigInteger, nullable=False, index=True)  # TARGET
    sale_date = Column(Date, nullable=False, index=True)
    sale_year = Column(Integer, index=True)  # Extracted for easy filtering
    
    # Spatial features (computed during ETL)
    h3_index = Column(String(20), index=True)  # H3 hex at resolution 8
    distance_to_center_km = Column(Float)  # Distance to Manhattan center
    
    # NYC administrative districts
    community_board = Column(Integer)
    council_district = Column(Integer)
    census_tract = Column(Float)
    nta = Column(String(10))  # Neighborhood Tabulation Area
    
    # BIN (Building Identification Number)
    bin_number = Column(BigInteger)  # NYC BIN
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_sales_borough_year', 'borough', 'sale_year'),
        Index('ix_sales_coords', 'latitude', 'longitude'),
        Index('ix_sales_price_range', 'sale_price', 'sale_year'),
    )
    
    def __repr__(self):
        return f"<Sales(bbl={self.bbl}, sale_price=${self.sale_price:,}, date={self.sale_date})>"


# =============================================================================
# Predictions Table
# =============================================================================

class Prediction(Base):
    """
    Model predictions with uncertainty bounds.
    
    Stores predictions for tracking and analysis.
    """
    __tablename__ = "predictions"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Link to property
    bbl = Column(BigInteger, nullable=False, index=True)
    
    # Prediction info
    predicted_price = Column(Float, nullable=False)
    prediction_lower = Column(Float)  # Lower bound (e.g., 10th percentile)
    prediction_upper = Column(Float)  # Upper bound (e.g., 90th percentile)
    
    # Actual (for validation)
    actual_price = Column(Float)
    
    # Error metrics (computed if actual available)
    absolute_error = Column(Float)
    percent_error = Column(Float)
    
    # Model info
    model_version = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), default="xgboost")
    
    # Metadata
    prediction_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<Prediction(bbl={self.bbl}, predicted=${self.predicted_price:,.0f})>"


# =============================================================================
# Model Performance Table
# =============================================================================

class ModelPerformance(Base):
    """
    Track model performance over time.
    
    One row per model version with aggregate metrics.
    """
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Model identification
    model_version = Column(String(50), nullable=False, unique=True)
    model_name = Column(String(100), default="xgboost")
    
    # Training info
    training_records = Column(Integer)
    test_records = Column(Integer)
    features_used = Column(Text)  # JSON list of features
    
    # Performance metrics
    r2_score = Column(Float)
    mae = Column(Float)  # Mean Absolute Error
    mdape = Column(Float)  # Median Absolute Percentage Error
    ppe10 = Column(Float)  # Percentage within ±10%
    ppe20 = Column(Float)  # Percentage within ±20%
    
    # Metadata
    trained_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    def __repr__(self):
        return f"<ModelPerformance(version={self.model_version}, PPE10={self.ppe10:.1%})>"


# =============================================================================
# Helper Functions
# =============================================================================

def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")


def drop_tables():
    """Drop all tables (use with caution!)."""
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All tables dropped")


def get_session():
    """Get a new database session."""
    return SessionLocal()


# =============================================================================
# Connection Test
# =============================================================================

def test_connection():
    """Test database connection."""
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database management")
    parser.add_argument("command", choices=["create", "drop", "test"])
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_tables()
    elif args.command == "drop":
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == "yes":
            drop_tables()
        else:
            print("Cancelled")
    elif args.command == "test":
        test_connection()
