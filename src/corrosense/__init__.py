"""
CorroSense AI - Electrochemical, Nernst-Planck & Flow Corrosion Intelligence
"""

from .kinetics import (
    calculate_tafel_corrosion,
    calculate_polarization_resistance
)
from .transport import (
    calculate_flow_corrosion_kinetics,
    HydrodynamicCorrosionResult
)
from .pitting import (
    evaluate_pitting_susceptibility,
    PittingResult
)
from .electrodiffusion import (
    simulate_nernst_planck_film,
    ElectrodiffusionResult
)
from .conformal import ConformalCorrosionUQ

__version__ = "1.0.0"
__all__ = [
    "calculate_tafel_corrosion",
    "calculate_polarization_resistance",
    "calculate_flow_corrosion_kinetics",
    "HydrodynamicCorrosionResult",
    "evaluate_pitting_susceptibility",
    "PittingResult",
    "simulate_nernst_planck_film",
    "ElectrodiffusionResult",
    "ConformalCorrosionUQ",
    "__version__"
]
