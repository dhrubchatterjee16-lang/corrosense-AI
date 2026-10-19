"""
Tests for Pitting Susceptibility & PREN
"""

import unittest
from corrosense.pitting import evaluate_pitting_susceptibility

class TestPitting(unittest.TestCase):
    def test_316_stainless_steel(self):
        # 316 SS: ~17% Cr, 2.5% Mo, 0.05% N -> PREN ~ 17 + 8.25 + 0.8 = ~26
        res = evaluate_pitting_susceptibility(cr_pct=17.0, mo_pct=2.5, n_pct=0.05, chloride_ppm=19000.0, temp_C=20.0)
        self.assertAlmostEqual(res.pren, 26.05, delta=0.5)
        self.assertEqual(res.alloy_class, "Duplex Stainless Steel (e.g., 2205)" if res.pren >= 25 else "Standard Austenitic (e.g., 304/316)")

    def test_super_duplex_2507(self):
        # 2507: 25% Cr, 4% Mo, 0.28% N -> PREN = 25 + 13.2 + 4.48 = 42.68
        res = evaluate_pitting_susceptibility(cr_pct=25.0, mo_pct=4.0, n_pct=0.28, chloride_ppm=19000.0, temp_C=50.0)
        self.assertGreaterEqual(res.pren, 40.0)
        self.assertEqual(res.alloy_class, "Super Duplex / Super Austenitic (e.g., 2507, 6Mo)")
        # High PREN gives high CPT
        self.assertGreater(res.cpt_celsius, 70.0)

    def test_temperature_risk_trigger(self):
        # Above CPT must trigger is_pitting_risk = True
        res_hot = evaluate_pitting_susceptibility(cr_pct=18.0, mo_pct=2.0, n_pct=0.05, temp_C=90.0)
        self.assertTrue(res_hot.is_pitting_risk)

        res_cold = evaluate_pitting_susceptibility(cr_pct=18.0, mo_pct=2.0, n_pct=0.05, temp_C=5.0)
        self.assertFalse(res_cold.is_pitting_risk)

if __name__ == "__main__":
    unittest.main()
