"""
Tests for Polarization Resistance & Tafel Kinetics
"""

import unittest
from corrosense.kinetics import calculate_polarization_resistance, calculate_tafel_corrosion

class TestKinetics(unittest.TestCase):
    def test_stern_geary_invariance(self):
        icorr = 0.05 # A/m^2
        res = calculate_polarization_resistance(icorr_A_m2=icorr, beta_a_V_dec=0.12, beta_c_V_dec=0.12)
        # B = 0.12 * 0.12 / (2.302585 * 0.24) = 0.0144 / 0.5526 = 0.02606 V
        self.assertAlmostEqual(res["stern_geary_B_V"], 0.02606, places=4)
        # Rp = B / icorr
        self.assertAlmostEqual(res["polarization_resistance_ohm_m2"], 0.02606 / 0.05, delta=0.01)

    def test_tafel_zero_overpotential(self):
        res = calculate_tafel_corrosion(overpotential_V=0.0, icorr_A_m2=0.05)
        # At zero overpotential, net current is zero
        self.assertAlmostEqual(res["net_current_density_A_m2"], 0.0, places=5)
        self.assertGreater(res["corrosion_rate_mmyr"], 0.0)

    def test_anodic_polarization(self):
        res = calculate_tafel_corrosion(overpotential_V=0.10, icorr_A_m2=0.05)
        # Positive overpotential must produce positive net anodic current
        self.assertGreater(res["net_current_density_A_m2"], 0.0)

if __name__ == "__main__":
    unittest.main()
