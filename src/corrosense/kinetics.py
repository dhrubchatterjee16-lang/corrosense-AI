"""
Electrochemical Kinetics & Stern-Geary Polarization Resistance
"""

import math
from typing import Dict, Any

FARADAY = 96485.3321  # C/mol

def calculate_polarization_resistance(
    icorr_A_m2: float,
    beta_a_V_dec: float = 0.12,
    beta_c_V_dec: float = 0.12
) -> Dict[str, float]:
    """
    Calculate Stern-Geary coefficient B and Polarization Resistance (R_p).
    B = (beta_a * beta_c) / (2.303 * (beta_a + beta_c))
    R_p = B / icorr
    """
    B = (beta_a_V_dec * beta_c_V_dec) / (math.log(10.0) * (beta_a_V_dec + beta_c_V_dec))
    Rp = B / max(1e-12, icorr_A_m2)
    return {
        "stern_geary_B_V": round(B, 5),
        "polarization_resistance_ohm_m2": round(Rp, 2)
    }

def calculate_tafel_corrosion(
    overpotential_V: float,
    icorr_A_m2: float = 0.05,
    beta_a_V_dec: float = 0.10,
    beta_c_V_dec: float = 0.12,
    equivalent_weight_g_eq: float = 27.92, # Fe (55.845 / 2)
    density_g_cm3: float = 7.87
) -> Dict[str, float]:
    """
    Calculate net current density using Butler-Volmer Tafel relation:
    i_net = i_corr * [ 10^(eta / beta_a) - 10^(-eta / beta_c) ]
    Converts i_corr to corrosion rate in mm/year.
    """
    term_a = 10.0 ** (overpotential_V / max(1e-4, beta_a_V_dec))
    term_c = 10.0 ** (-overpotential_V / max(1e-4, beta_c_V_dec))
    i_net = icorr_A_m2 * (term_a - term_c)

    # Faraday's law conversion to mm/year
    # mm/yr = (i_corr (A/m^2) * EW (g/mol/e-) * 31536000 s/yr) / (F (C/mol) * density (g/cm^3) * 10000 cm^2/m^2) * 10 mm/cm
    # = (i_corr * EW * 3.1536e7) / (96485.33 * density * 1e3)
    corrosion_rate_mmyr = (icorr_A_m2 * equivalent_weight_g_eq * 3.1536e7) / (
        FARADAY * density_g_cm3 * 1000.0
    )

    sg = calculate_polarization_resistance(icorr_A_m2, beta_a_V_dec, beta_c_V_dec)

    return {
        "overpotential_V": overpotential_V,
        "net_current_density_A_m2": round(i_net, 6),
        "icorr_A_m2": round(icorr_A_m2, 6),
        "corrosion_rate_mmyr": round(corrosion_rate_mmyr, 5),
        "polarization_resistance_ohm_m2": sg["polarization_resistance_ohm_m2"]
    }
