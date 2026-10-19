"""
Tests for Hydrodynamic Flow-Accelerated Corrosion
"""

import unittest
from corrosense.transport import calculate_flow_corrosion_kinetics

class TestTransport(unittest.TestCase):
    def test_turbulent_regime_and_scaling(self):
        # Velocity = 2.0 m/s in 50 mm pipe -> Re > 2300 (Turbulent)
        res = calculate_flow_corrosion_kinetics(fluid_velocity_m_s=2.0, pipe_diameter_m=0.05)
        self.assertGreater(res.reynolds_number, 2300.0)
        self.assertEqual(res.flow_regime, "Turbulent")
        self.assertGreater(res.limiting_current_density_A_m2, 0.0)
        self.assertGreater(res.flow_accelerated_corrosion_mmyr, 0.0)

    def test_laminar_regime(self):
        # Very low velocity = 0.01 m/s -> Re < 2300
        res = calculate_flow_corrosion_kinetics(fluid_velocity_m_s=0.01, pipe_diameter_m=0.02)
        self.assertLess(res.reynolds_number, 2300.0)
        self.assertEqual(res.flow_regime, "Laminar")
        self.assertGreater(res.sherwood_number, 3.0)

    def test_velocity_corrosion_monotonicity(self):
        res1 = calculate_flow_corrosion_kinetics(fluid_velocity_m_s=1.0)
        res2 = calculate_flow_corrosion_kinetics(fluid_velocity_m_s=3.0)
        # Higher velocity must increase mass transfer and limiting current
        self.assertGreater(res2.limiting_current_density_A_m2, res1.limiting_current_density_A_m2)
        self.assertGreater(res2.flow_accelerated_corrosion_mmyr, res1.flow_accelerated_corrosion_mmyr)

if __name__ == "__main__":
    unittest.main()
