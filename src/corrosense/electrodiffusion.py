"""
Nernst-Planck Electrodiffusion Flux Across Passive Films
=========================================================
Solves 1D Nernst-Planck transport equation:
  J_i = - D_i * (dc_i / dx) - (z_i * D_i * F / (R * T)) * c_i * (dphi / dx)
Ensuring flux conservation and charge balance across passive boundary layers.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Any

R_GAS = 8.314462618   # J/(mol*K)
FARADAY = 96485.3321  # C/mol

@dataclass
class ElectrodiffusionResult:
    species: str
    diffusional_flux_mol_m2_s: float
    migrational_flux_mol_m2_s: float
    total_flux_mol_m2_s: float
    current_density_A_m2: float

def simulate_nernst_planck_film(
    species: str = "Fe2+",
    valence_z: int = 2,
    c_metal_mol_m3: float = 100.0,
    c_solution_mol_m3: float = 0.5,
    film_thickness_nm: float = 5.0,
    potential_drop_V: float = 0.35,
    temp_K: float = 298.15,
    diffusivity_m2_s: float = 1.0e-14
) -> ElectrodiffusionResult:
    """
    Compute diffusional and electromigrational fluxes across a passive oxide film or diffusion layer.
    """
    delta_x_m = max(1e-10, film_thickness_nm * 1e-9)
    grad_c = (c_solution_mol_m3 - c_metal_mol_m3) / delta_x_m
    grad_phi = potential_drop_V / delta_x_m
    c_avg = 0.5 * (c_metal_mol_m3 + c_solution_mol_m3)

    # Diffusive flux: J_diff = - D * grad(c)
    j_diff = - diffusivity_m2_s * grad_c

    # Migrational flux: J_migr = - (z * D * F / (R * T)) * c_avg * grad(phi)
    u_i = (diffusivity_m2_s * FARADAY) / (R_GAS * temp_K)
    j_migr = - valence_z * u_i * c_avg * grad_phi

    j_total = j_diff + j_migr
    i_density = valence_z * FARADAY * j_total

    return ElectrodiffusionResult(
        species=species,
        diffusional_flux_mol_m2_s=float(j_diff),
        migrational_flux_mol_m2_s=float(j_migr),
        total_flux_mol_m2_s=float(j_total),
        current_density_A_m2=float(i_density)
    )
