"""
Tests for Nernst-Planck Electrodiffusion Flux
"""

import unittest
from corrosense.electrodiffusion import simulate_nernst_planck_film

class TestElectrodiffusion(unittest.TestCase):
    def test_nernst_planck_conservation(self):
        res = simulate_nernst_planck_film(
            species="Fe2+",
            valence_z=2,
            c_metal_mol_m3=100.0,
            c_solution_mol_m3=0.5,
            film_thickness_nm=5.0,
            potential_drop_V=0.25
        )
        # Verify total flux equals sum of diffusional and migrational components
        self.assertAlmostEqual(
            res.total_flux_mol_m2_s,
            res.diffusional_flux_mol_m2_s + res.migrational_flux_mol_m2_s,
            places=9
        )
        # Current density = z * F * J_total
        expected_current = 2.0 * 96485.3321 * res.total_flux_mol_m2_s
        self.assertAlmostEqual(res.current_density_A_m2, expected_current, delta=1e-5)

if __name__ == "__main__":
    unittest.main()
