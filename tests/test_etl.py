import unittest

import numpy as np
import pandas as pd

from src.etl import clean_data, create_property_id, deduplicate, enrich_sales_history, impute_missing_values


class TestEtlTransforms(unittest.TestCase):
    def test_clean_data_filters_invalid_rows(self):
        raw = pd.DataFrame(
            [
                {
                    "sale_price": "500000",
                    "building_class_category": "01 ONE FAMILY DWELLINGS",
                    "building_class_at_time_of": "A1",
                    "latitude": 40.75,
                    "longitude": -73.98,
                    "bbl": 1000000001,
                    "sale_date": "2024-01-15",
                    "year_built": 1990,
                },
                {
                    "sale_price": "5000",
                    "building_class_category": "01 ONE FAMILY DWELLINGS",
                    "building_class_at_time_of": "A1",
                    "latitude": 40.75,
                    "longitude": -73.98,
                    "bbl": 1000000002,
                    "sale_date": "2024-01-15",
                    "year_built": 1990,
                },
                {
                    "sale_price": "750000",
                    "building_class_category": "31 COMMERCIAL",
                    "building_class_at_time_of": "K1",
                    "latitude": 40.75,
                    "longitude": -73.98,
                    "bbl": 1000000003,
                    "sale_date": "2024-01-15",
                    "year_built": 1980,
                },
                {
                    "sale_price": "650000",
                    "building_class_category": "01 ONE FAMILY DWELLINGS",
                    "building_class_at_time_of": "A1",
                    "latitude": np.nan,
                    "longitude": -73.98,
                    "bbl": 1000000004,
                    "sale_date": "2024-01-15",
                    "year_built": 2000,
                },
                {
                    "sale_price": "650000",
                    "building_class_category": "01 ONE FAMILY DWELLINGS",
                    "building_class_at_time_of": "A1",
                    "latitude": 40.75,
                    "longitude": -73.98,
                    "bbl": np.nan,
                    "sale_date": "2024-01-15",
                    "year_built": 2000,
                },
                {
                    "sale_price": "800000",
                    "building_class_category": "01 ONE FAMILY DWELLINGS",
                    "building_class_at_time_of": "A1",
                    "latitude": 40.75,
                    "longitude": -73.98,
                    "bbl": 0,
                    "sale_date": "2024-01-15",
                    "year_built": 1990,
                },
            ]
        )

        cleaned = clean_data(raw)
        self.assertEqual(len(cleaned), 1)
        self.assertIn("building_class", cleaned.columns)
        self.assertEqual(int(cleaned.iloc[0]["sale_price"]), 500000)
        self.assertEqual(int(cleaned.iloc[0]["sale_year"]), 2024)

    def test_property_id_and_deduplicate(self):
        df = pd.DataFrame(
            [
                {"bbl": 1000000001, "apartment_number": "1A", "sale_date": "2023-01-01", "sale_price": 100000},
                {"bbl": 1000000001, "apartment_number": "1A", "sale_date": "2023-01-01", "sale_price": 100000},
                {"bbl": 1000000001, "apartment_number": "1A", "sale_date": "2024-01-01", "sale_price": 120000},
                {"bbl": 1000000001, "apartment_number": "2B", "sale_date": "2023-01-01", "sale_price": 100000},
                {"bbl": 1000000002, "apartment_number": np.nan, "sale_date": "2024-06-01", "sale_price": 90000},
            ]
        )
        df["sale_date"] = pd.to_datetime(df["sale_date"])

        with_ids = create_property_id(df)
        deduped = deduplicate(with_ids)

        self.assertEqual(len(deduped), 4)
        self.assertIn("1000000001_1A", deduped["property_id"].values)
        self.assertIn("1000000002", deduped["property_id"].values)

        apt_1a = deduped[deduped["property_id"] == "1000000001_1A"].sort_values("sale_date")
        self.assertEqual(len(apt_1a), 2)

    def test_enrich_sales_history_derivations(self):
        df = pd.DataFrame(
            [
                {"property_id": "1000000001_1A", "sale_date": "2023-01-01", "sale_price": 100000},
                {"property_id": "1000000001_1A", "sale_date": "2024-01-01", "sale_price": 150000},
                {"property_id": "1000000002", "sale_date": "2024-06-01", "sale_price": 90000},
            ]
        )
        df["sale_date"] = pd.to_datetime(df["sale_date"])

        enriched = enrich_sales_history(df)
        first = enriched[enriched["property_id"] == "1000000001_1A"].sort_values("sale_date").iloc[0]
        second = enriched[enriched["property_id"] == "1000000001_1A"].sort_values("sale_date").iloc[1]

        self.assertEqual(int(first["sale_sequence"]), 1)
        self.assertEqual(int(second["sale_sequence"]), 2)
        self.assertFalse(bool(first["is_latest_sale"]))
        self.assertTrue(bool(second["is_latest_sale"]))
        self.assertEqual(float(second["previous_sale_price"]), 100000.0)
        self.assertAlmostEqual(float(second["price_change_pct"]), 50.0, places=2)

    def test_imputation_flags_and_fallback_levels(self):
        df = pd.DataFrame(
            [
                {"neighborhood": "N1", "building_class": "C1", "borough": 1, "gross_square_feet": 1000, "year_built": 1990},
                {"neighborhood": "N1", "building_class": "C1", "borough": 1, "gross_square_feet": 0, "year_built": 0},
                {"neighborhood": "N2", "building_class": "C2", "borough": 2, "gross_square_feet": 2000, "year_built": 1980},
                {"neighborhood": "N3", "building_class": "C2", "borough": 2, "gross_square_feet": np.nan, "year_built": 0},
                {"neighborhood": "N4", "building_class": "C3", "borough": 3, "gross_square_feet": 3000, "year_built": 1970},
                {"neighborhood": "N5", "building_class": "C3", "borough": 4, "gross_square_feet": 0, "year_built": 0},
                {"neighborhood": "N6", "building_class": "C4", "borough": 5, "gross_square_feet": 0, "year_built": 0},
            ]
        )

        out = impute_missing_values(df)

        self.assertTrue(bool(out.iloc[1]["sqft_imputed"]))
        self.assertEqual(out.iloc[1]["sqft_imputation_level"], "neighborhood_class")
        self.assertEqual(float(out.iloc[1]["gross_square_feet"]), 1000.0)

        self.assertEqual(out.iloc[3]["sqft_imputation_level"], "borough_class")
        self.assertEqual(float(out.iloc[3]["gross_square_feet"]), 2000.0)

        self.assertEqual(out.iloc[5]["sqft_imputation_level"], "class_only")
        self.assertEqual(float(out.iloc[5]["gross_square_feet"]), 3000.0)

        self.assertEqual(out.iloc[6]["sqft_imputation_level"], "citywide")
        self.assertEqual(float(out.iloc[6]["gross_square_feet"]), 2000.0)

        self.assertTrue(bool(out.iloc[6]["year_built_imputed"]))
        self.assertEqual(int(out.iloc[6]["year_built"]), 1980)


if __name__ == "__main__":
    unittest.main()
