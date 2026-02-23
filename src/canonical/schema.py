from __future__ import annotations

CANONICAL_REQUIRED_COLUMNS = [
    "sale_date",
    "sale_price",
    "latitude",
    "longitude",
    "property_segment",
    "gross_square_feet",
    "year_built",
]

CANONICAL_OPTIONAL_COLUMNS = [
    "bbl",
    "borough",
    "block",
    "lot",
    "building_class",
    "building_class_category",
    "h3_index",
    "distance_to_center_km",
    "price_tier",
    "neighborhood",
    "residential_units",
    "total_units",
]

CANONICAL_COLUMN_TYPES = {
    "sale_date": "datetime64[ns]",
    "sale_price": "float64",
    "latitude": "float64",
    "longitude": "float64",
    "gross_square_feet": "float64",
    "year_built": "int64",
}
