"""
Pitting Corrosion Susceptibility & ASTM G48 Thermochemical Analysis
"""

import math
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PittingResult:
    pren: float
    cpt_celsius: float
    effective_cpt_celsius: float
    is_pitting_risk: bool
    alloy_class: str
    chloride_ppm: float
    temperature_C: float

def evaluate_pitting_susceptibility(
    cr_pct: float = 18.0,
    mo_pct: float = 2.5,
    n_pct: float = 0.15,
    chloride_ppm: float = 19000.0,
    temp_C: float = 30.0
) -> PittingResult:
    """
    Calculate Pitting Resistance Equivalent Number (PREN) and Critical Pitting Temperature (CPT).
    Formula: PREN = %Cr + 3.3*%Mo + 16*%N
    Empirical CPT (ASTM G48): CPT (°C) = 2.5 * PREN - 31.0 - chloride_penalty
    """
    pren = cr_pct + 3.3 * mo_pct + 16.0 * n_pct
    cpt_base = 2.5 * pren - 31.0
    
    # Chloride chemical activity penalty
    cl_penalty = math.log10(max(10.0, chloride_ppm) / 1000.0) * 4.0
    cpt_effective = cpt_base - cl_penalty

    is_risk = temp_C >= cpt_effective

    if pren < 25.0:
        alloy_class = "Standard Austenitic (e.g., 304/316)"
    elif pren < 40.0:
        alloy_class = "Duplex Stainless Steel (e.g., 2205)"
    else:
        alloy_class = "Super Duplex / Super Austenitic (e.g., 2507, 6Mo)"

    return PittingResult(
        pren=round(pren, 2),
        cpt_celsius=round(cpt_base, 2),
        effective_cpt_celsius=round(cpt_effective, 2),
        is_pitting_risk=is_risk,
        alloy_class=alloy_class,
        chloride_ppm=chloride_ppm,
        temperature_C=temp_C
    )
