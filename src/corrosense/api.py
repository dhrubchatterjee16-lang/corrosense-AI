"""
FastAPI Microservice for CorroSense AI
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .transport import calculate_flow_corrosion_kinetics
from .pitting import evaluate_pitting_susceptibility
from .kinetics import calculate_tafel_corrosion, calculate_polarization_resistance
from .electrodiffusion import simulate_nernst_planck_film
from .conformal import ConformalCorrosionUQ

if HAS_FASTAPI:
    app = FastAPI(
        title="CorroSense AI REST API",
        description="Electrochemical, Nernst-Planck & Flow Corrosion Intelligence Microservice",
        version="1.0.0"
    )

    class FlowCorrosionRequest(BaseModel):
        fluid_velocity_m_s: float = Field(1.5, ge=0.01, le=50.0)
        pipe_diameter_m: float = Field(0.05, ge=0.001, le=5.0)
        temp_C: float = Field(25.0, ge=-10.0, le=200.0)
        dissolved_o2_ppm: float = Field(8.0, ge=0.0, le=100.0)

    class PittingRequest(BaseModel):
        cr_pct: float = Field(18.0, ge=0.0, le=50.0)
        mo_pct: float = Field(2.5, ge=0.0, le=20.0)
        n_pct: float = Field(0.15, ge=0.0, le=5.0)
        chloride_ppm: float = Field(19000.0, ge=0.0, le=250000.0)
        temp_C: float = Field(30.0, ge=-20.0, le=150.0)

    @app.get("/")
    def root():
        return {
            "service": "CorroSense AI",
            "version": "1.0.0",
            "status": "operational"
        }

    @app.post("/corrosion/flow")
    def api_flow(req: FlowCorrosionRequest):
        res = calculate_flow_corrosion_kinetics(
            fluid_velocity_m_s=req.fluid_velocity_m_s,
            pipe_diameter_m=req.pipe_diameter_m,
            temp_C=req.temp_C,
            dissolved_o2_ppm=req.dissolved_o2_ppm
        )
        uq = ConformalCorrosionUQ().predict_interval(res.flow_accelerated_corrosion_mmyr)
        return {
            "hydrodynamics": {
                "reynolds_number": res.reynolds_number,
                "schmidt_number": res.schmidt_number,
                "sherwood_number": res.sherwood_number,
                "limiting_current_A_m2": res.limiting_current_density_A_m2,
                "corrosion_rate_mmyr": res.flow_accelerated_corrosion_mmyr,
                "flow_regime": res.flow_regime
            },
            "conformal_uq": uq
        }

    @app.post("/corrosion/pitting")
    def api_pitting(req: PittingRequest):
        res = evaluate_pitting_susceptibility(
            cr_pct=req.cr_pct,
            mo_pct=req.mo_pct,
            n_pct=req.n_pct,
            chloride_ppm=req.chloride_ppm,
            temp_C=req.temp_C
        )
        return {
            "pren": res.pren,
            "cpt_celsius": res.cpt_celsius,
            "effective_cpt_celsius": res.effective_cpt_celsius,
            "is_pitting_risk": res.is_pitting_risk,
            "alloy_class": res.alloy_class,
            "chloride_ppm": res.chloride_ppm,
            "temperature_C": res.temperature_C
        }
else:
    app = None
