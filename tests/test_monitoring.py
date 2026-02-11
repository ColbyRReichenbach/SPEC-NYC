import unittest

import numpy as np
import pandas as pd

from src.monitoring.drift import calculate_ks, calculate_psi, monitor_drift
from src.monitoring.performance import evaluate_performance
from src.retrain_policy import evaluate_retrain_policy


class TestMonitoring(unittest.TestCase):
    def test_psi_and_ks_detect_shift(self):
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.normal(0, 1, 1000))
        cur = pd.Series(rng.normal(1.2, 1, 1000))
        psi = calculate_psi(ref, cur, bins=10)
        ks = calculate_ks(ref, cur)
        self.assertGreater(psi, 0.1)
        self.assertGreater(ks, 0.1)

    def test_monitor_drift_table(self):
        ref = pd.DataFrame({"gross_square_feet": np.random.normal(1000, 200, 400)})
        cur = pd.DataFrame({"gross_square_feet": np.random.normal(1400, 200, 400)})
        out = monitor_drift(ref, cur, features=["gross_square_feet"])
        self.assertEqual(len(out), 1)
        self.assertIn(out.iloc[0]["status"], {"warn", "alert", "ok"})

    def test_performance_status_alert(self):
        df = pd.DataFrame(
            {
                "sale_price": [100] * 40 + [200] * 40,
                "predicted_price": [180] * 40 + [80] * 40,
                "property_segment": ["A"] * 40 + ["B"] * 40,
                "price_tier": ["entry"] * 80,
            }
        )
        report = evaluate_performance(df, warn_ppe10=0.75, critical_ppe10=0.65, warn_mdape=0.08, critical_mdape=0.12)
        self.assertEqual(report["status"], "alert")

    def test_retrain_policy_decision(self):
        metrics = {
            "overall": {"ppe10": 0.70, "mdape": 0.09, "r2": 0.7},
            "metadata": {"trained_at_utc": "2025-01-01T00:00:00"},
        }
        perf = {"status": "warn"}
        drift_df = pd.DataFrame([{"feature": "x", "status": "alert"}])
        decision = evaluate_retrain_policy(
            metrics_json=metrics,
            performance_json=perf,
            drift_df=drift_df,
            max_model_age_days=30,
            min_ppe10=0.75,
            max_mdape=0.08,
            max_drift_alerts=0,
        )
        self.assertTrue(decision["should_retrain"])
        self.assertEqual(decision["decision"], "retrain")


if __name__ == "__main__":
    unittest.main()

