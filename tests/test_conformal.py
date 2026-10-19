"""
Tests for Conformal Prediction Uncertainty Quantification
"""

import unittest
from corrosense.conformal import ConformalCorrosionUQ

class TestConformal(unittest.TestCase):
    def test_coverage_ordering(self):
        uq = ConformalCorrosionUQ()
        res90 = uq.predict_interval(predicted_corrosion_rate_mmyr=0.25, confidence=0.90)
        res95 = uq.predict_interval(predicted_corrosion_rate_mmyr=0.25, confidence=0.95)

        self.assertLessEqual(res90["lower_bound_mmyr"], 0.25)
        self.assertGreaterEqual(res90["upper_bound_mmyr"], 0.25)

        # 95% confidence interval must be wider than or equal to 90%
        self.assertGreaterEqual(res95["interval_width_mmyr"], res90["interval_width_mmyr"])

    def test_non_negativity(self):
        uq = ConformalCorrosionUQ()
        # Even with very small prediction, lower bound cannot be negative
        res = uq.predict_interval(predicted_corrosion_rate_mmyr=0.01, confidence=0.90)
        self.assertGreaterEqual(res["lower_bound_mmyr"], 0.0)

if __name__ == "__main__":
    unittest.main()
