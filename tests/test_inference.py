import unittest

import numpy as np
import pandas as pd

from src.inference import predict_dataframe, predict_single_row


class _ConstantPipeline:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)


class TestInference(unittest.TestCase):
    def test_predict_dataframe_global(self):
        artifact = {
            "model_strategy": "global",
            "feature_columns": ["f1", "f2"],
            "pipeline": _ConstantPipeline(111.0),
        }
        frame = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "property_segment": ["A", "B"]})
        preds, routes = predict_dataframe(artifact, frame)
        self.assertEqual(preds.tolist(), [111.0, 111.0])
        self.assertEqual(routes.tolist(), ["global", "global"])

    def test_predict_dataframe_segmented_router_with_fallback(self):
        artifact = {
            "model_strategy": "segmented_router",
            "feature_columns": ["f1", "f2"],
            "fallback_pipeline": _ConstantPipeline(200.0),
            "segment_pipelines": {
                "SINGLE_FAMILY": _ConstantPipeline(300.0),
            },
            "router_column": "property_segment",
        }
        frame = pd.DataFrame(
            {
                "f1": [1, 2, 3],
                "f2": [4, 5, 6],
                "property_segment": ["SINGLE_FAMILY", "ELEVATOR", None],
            }
        )
        preds, routes = predict_dataframe(artifact, frame)
        self.assertEqual(preds.tolist(), [300.0, 200.0, 200.0])
        self.assertEqual(routes.tolist(), ["route:SINGLE_FAMILY", "fallback_global", "fallback_global"])

    def test_predict_single_row_segmented_router(self):
        artifact = {
            "model_strategy": "segmented_router",
            "feature_columns": ["f1"],
            "fallback_pipeline": _ConstantPipeline(150.0),
            "segment_pipelines": {"WALKUP": _ConstantPipeline(175.0)},
            "router_column": "property_segment",
        }
        row = pd.Series({"f1": 10, "property_segment": "WALKUP"})
        pred, route = predict_single_row(artifact, row)
        self.assertEqual(pred, 175.0)
        self.assertEqual(route, "route:WALKUP")

    def test_predict_dataframe_segment_plus_tier_routes(self):
        artifact = {
            "model_strategy": "segmented_router",
            "feature_columns": ["f1"],
            "fallback_pipeline": _ConstantPipeline(90.0),
            "segment_pipelines": {
                "SINGLE_FAMILY||entry": _ConstantPipeline(120.0),
                "SINGLE_FAMILY||luxury": _ConstantPipeline(180.0),
            },
            "router_columns": ["property_segment", "price_tier_proxy"],
            "router_column": "property_segment",
        }
        frame = pd.DataFrame(
            {
                "f1": [1, 2, 3],
                "property_segment": ["SINGLE_FAMILY", "SINGLE_FAMILY", "ELEVATOR"],
                "price_tier_proxy": ["entry", "luxury", "entry"],
            }
        )
        preds, routes = predict_dataframe(artifact, frame)
        self.assertEqual(preds.tolist(), [120.0, 180.0, 90.0])
        self.assertEqual(
            routes.tolist(),
            ["route:SINGLE_FAMILY||entry", "route:SINGLE_FAMILY||luxury", "fallback_global"],
        )

    def test_predict_dataframe_segment_plus_tier_derives_proxy_at_inference_time(self):
        artifact = {
            "model_strategy": "segmented_router",
            "feature_columns": ["f1"],
            "fallback_pipeline": _ConstantPipeline(90.0),
            "segment_pipelines": {
                "SINGLE_FAMILY||entry": _ConstantPipeline(120.0),
                "SINGLE_FAMILY||luxury": _ConstantPipeline(180.0),
            },
            "router_columns": ["property_segment", "price_tier_proxy"],
            "router_column": "property_segment",
            "price_tier_proxy_bins": {
                "version": 1,
                "global": {"q25": 1.0, "q50": 2.0, "q75": 3.0, "n": 10},
                "segments": {
                    "SINGLE_FAMILY": {"q25": 1.0, "q50": 2.0, "q75": 3.0, "n": 10},
                },
            },
        }
        frame = pd.DataFrame(
            {
                "f1": [1, 2],
                "property_segment": ["SINGLE_FAMILY", "SINGLE_FAMILY"],
                "gross_square_feet": [1.0, 10_000.0],
                "building_age": [90, 1],
                "distance_to_center_km": [20.0, 0.2],
                "total_units": [1, 4],
                "residential_units": [1, 4],
                "borough": [3, 1],
            }
        )
        preds, routes = predict_dataframe(artifact, frame)
        self.assertEqual(preds.tolist(), [120.0, 180.0])
        self.assertEqual(routes.tolist(), ["route:SINGLE_FAMILY||entry", "route:SINGLE_FAMILY||luxury"])

    def test_predict_dataframe_disallows_target_derived_price_tier_routing(self):
        artifact = {
            "model_strategy": "segmented_router",
            "feature_columns": ["f1"],
            "fallback_pipeline": _ConstantPipeline(90.0),
            "segment_pipelines": {"SINGLE_FAMILY||entry": _ConstantPipeline(120.0)},
            "router_columns": ["property_segment", "price_tier"],
        }
        frame = pd.DataFrame({"f1": [1], "property_segment": ["SINGLE_FAMILY"], "price_tier": ["entry"]})
        with self.assertRaises(ValueError):
            predict_dataframe(artifact, frame)

    def test_predict_dataframe_missing_required_features_raises(self):
        artifact = {
            "model_strategy": "global",
            "feature_columns": ["f1", "f2"],
            "pipeline": _ConstantPipeline(111.0),
        }
        frame = pd.DataFrame({"f1": [1]})
        with self.assertRaises(ValueError):
            predict_dataframe(artifact, frame)

    def test_predict_dataframe_derives_temporal_features_from_sale_date(self):
        artifact = {
            "model_strategy": "global",
            "feature_columns": ["days_since_2019_start", "month_sin", "month_cos", "rate_regime_bucket"],
            "pipeline": _ConstantPipeline(111.0),
        }
        frame = pd.DataFrame({"sale_date": ["2024-02-15", "2024-08-20"]})
        preds, routes = predict_dataframe(artifact, frame)
        self.assertEqual(preds.tolist(), [111.0, 111.0])
        self.assertEqual(routes.tolist(), ["global", "global"])


if __name__ == "__main__":
    unittest.main()
