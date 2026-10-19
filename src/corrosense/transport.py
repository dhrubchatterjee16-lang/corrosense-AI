"""
Hydrodynamic & Mass-Transfer Limited Corrosion Modeling
"""

import math
from dataclasses import dataclass
from typing import Dict, Any

R_GAS = 8.314462618     # J/(mol*K)
FARADAY = 96485.3321    # C/mol
WATER_DENSITY = 1000.0  # kg/m^3
WATER_VISCOSITY = 1.0e-3 # Pa*s
D_OXYGEN = 2.0e-9       # m^2/s

@dataclass
class HydrodynamicCorrosionResult:
    reynolds_number: float
    schmidt_number: float
    sherwood_number: float
    limiting_current_density_A_m2: float
    flow_accelerated_corrosion_mmyr: float
    flow_regime: str

def calculate_flow_corrosion_kinetics(
    fluid_velocity_m_s: float = 1.5,
    pipe_diameter_m: float = 0.05,
    temp_C: float = 25.0,
    dissolved_o2_ppm: float = 8.0,
    ph: float = 7.0
) -> HydrodynamicCorrosionResult:
    """
    Calculate dimensionless numbers and mass-transfer limited corrosion rate.
    Uses Berger-Hau turbulent Sherwood correlation and Faraday's law of electrolysis.
    """
    temp_K = temp_C + 273.15
    nu = (WATER_VISCOSITY / WATER_DENSITY) * math.exp(-0.015 * (temp_C - 25.0))
    D_eff = D_OXYGEN * (temp_K / 298.15) * (1.0e-3 / max(1e-4, nu * WATER_DENSITY))

    Re = max(1.0, (fluid_velocity_m_s * pipe_diameter_m) / max(1e-7, nu))
    Sc = max(1.0, nu / max(1e-12, D_eff))

    if Re > 2300:
        Sh = 0.023 * (Re ** 0.8) * (Sc ** 0.33)
        regime = "Turbulent"
    else:
        Sh = 3.66 + (0.0668 * (pipe_diameter_m / 1.0) * Re * Sc) / (
            1.0 + 0.04 * ((pipe_diameter_m / 1.0) * Re * Sc) ** 0.66
        )
        regime = "Laminar"

    k_m = (Sh * D_eff) / max(1e-4, pipe_diameter_m)
    C_bulk = (dissolved_o2_ppm * 1e-3) / 32.0 * WATER_DENSITY

    # 4e- oxygen reduction: O2 + 2H2O + 4e- -> 4OH-
    n_electrons = 4.0
    i_L = n_electrons * FARADAY * k_m * C_bulk

    # Conversion to mm/year for carbon steel (Fe -> Fe2+ + 2e-)
    corrosion_rate_mmyr = (i_L / (2.0 * FARADAY)) * (55.845 / 7870.0) * (365.25 * 86400.0) * 1000.0

    return HydrodynamicCorrosionResult(
        reynolds_number=round(Re, 1),
        schmidt_number=round(Sc, 1),
        sherwood_number=round(Sh, 1),
        limiting_current_density_A_m2=round(i_L, 4),
        flow_accelerated_corrosion_mmyr=round(corrosion_rate_mmyr, 4),
        flow_regime=regime
    )
