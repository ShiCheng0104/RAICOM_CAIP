from __future__ import annotations

import unittest

import numpy as np

from fraudsim.threshold_sandbox import evaluate_thresholds


class ThresholdSandboxTests(unittest.TestCase):
    def test_reports_decisions_and_high_risk_metrics(self) -> None:
        result = evaluate_thresholds(
            labels=np.array([0, 0, 1, 1]),
            scores=np.array([0.1, 0.6, 0.7, 0.9]),
            medium_threshold=0.5,
            high_threshold=0.8,
        )
        self.assertEqual(result["decisions"], {"pass": 1, "review": 2, "reject": 1})
        self.assertEqual(result["high_risk_metrics"]["true_positive"], 1)
        self.assertEqual(result["high_risk_metrics"]["false_positive"], 0)
        self.assertAlmostEqual(result["high_risk_metrics"]["recall"], 0.5)

    def test_rejects_invalid_threshold_order(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_thresholds(np.array([0]), np.array([0.2]), 0.8, 0.5)


if __name__ == "__main__":
    unittest.main()
