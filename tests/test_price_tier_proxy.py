import unittest

import pandas as pd

from src.price_tier_proxy import assign_price_tier_proxy, fit_price_tier_proxy_bins


class TestPriceTierProxy(unittest.TestCase):
    def test_proxy_is_not_target_derived(self):
        base = pd.DataFrame(
            {
                "property_segment": ["SINGLE_FAMILY", "SINGLE_FAMILY", "ELEVATOR", "ELEVATOR"],
                "gross_square_feet": [900, 2200, 700, 1800],
                "building_age": [75, 12, 90, 20],
                "distance_to_center_km": [8.0, 2.5, 10.0, 3.0],
                "total_units": [1, 2, 18, 40],
                "residential_units": [1, 2, 16, 35],
                "borough": [3, 1, 3, 1],
                "sale_price": [200_000, 1_900_000, 250_000, 2_200_000],
                "price_tier": ["entry", "luxury", "entry", "luxury"],
            }
        )
        altered_targets = base.copy()
        altered_targets["sale_price"] = [5_000_000, 10_000, 7_500_000, 20_000]
        altered_targets["price_tier"] = ["luxury", "entry", "luxury", "entry"]

        out_a, bins = assign_price_tier_proxy(base, min_segment_rows=1)
        out_b, _ = assign_price_tier_proxy(altered_targets, bins=bins, min_segment_rows=1)

        self.assertEqual(out_a["price_tier_proxy"].astype(str).tolist(), out_b["price_tier_proxy"].astype(str).tolist())
        self.assertEqual(out_a["price_tier_proxy_source"].astype(str).tolist(), out_b["price_tier_proxy_source"].astype(str).tolist())

    def test_sparse_segment_uses_global_fallback_bins(self):
        df = pd.DataFrame(
            {
                "property_segment": ["SINGLE_FAMILY"] * 6 + ["TINY_SEG"],
                "gross_square_feet": [900, 1000, 1100, 2000, 2200, 2500, 950],
                "building_age": [80, 70, 65, 15, 12, 8, 40],
                "distance_to_center_km": [9.0, 8.5, 8.2, 2.8, 2.5, 2.1, 7.2],
                "total_units": [1, 1, 1, 2, 2, 3, 1],
                "residential_units": [1, 1, 1, 2, 2, 3, 1],
                "borough": [3, 3, 3, 1, 1, 1, 4],
            }
        )
        bins = fit_price_tier_proxy_bins(df, min_segment_rows=3)
        out, _ = assign_price_tier_proxy(df, bins=bins, min_segment_rows=3)

        self.assertNotIn("TINY_SEG", bins["segments"])
        tiny_row = out[out["property_segment"] == "TINY_SEG"].iloc[0]
        self.assertEqual(str(tiny_row["price_tier_proxy_source"]), "global_fallback")


if __name__ == "__main__":
    unittest.main()
