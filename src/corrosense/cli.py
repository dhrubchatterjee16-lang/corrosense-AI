"""
Command-Line Interface for CorroSense AI
"""

import argparse
import sys
import json
from .transport import calculate_flow_corrosion_kinetics
from .pitting import evaluate_pitting_susceptibility
from .kinetics import calculate_tafel_corrosion
from .conformal import ConformalCorrosionUQ

def main():
    parser = argparse.ArgumentParser(
        prog="corrosense",
        description="CorroSense AI: Electrochemical, Nernst-Planck & Flow Corrosion Intelligence"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # flow subcommand
    flow_parser = subparsers.add_parser("flow", help="Calculate flow-accelerated corrosion kinetics")
    flow_parser.add_argument("--velocity", type=float, default=1.5, help="Fluid velocity (m/s)")
    flow_parser.add_argument("--diameter", type=float, default=0.05, help="Pipe diameter (m)")
    flow_parser.add_argument("--temp", type=float, default=25.0, help="Temperature in °C")
    flow_parser.add_argument("--o2", type=float, default=8.0, help="Dissolved O2 in ppm")

    # pitting subcommand
    pitting_parser = subparsers.add_parser("pitting", help="Evaluate localized pitting susceptibility & PREN")
    pitting_parser.add_argument("--cr", type=float, default=18.0, help="Chromium wt%")
    pitting_parser.add_argument("--mo", type=float, default=2.5, help="Molybdenum wt%")
    pitting_parser.add_argument("--n", type=float, default=0.15, help="Nitrogen wt%")
    pitting_parser.add_argument("--chloride", type=float, default=19000.0, help="Chloride ppm")
    pitting_parser.add_argument("--temp", type=float, default=30.0, help="Temperature in °C")

    # predict subcommand
    pred_parser = subparsers.add_parser("predict", help="Predict corrosion rate with conformal UQ")
    pred_parser.add_argument("--velocity", type=float, default=2.0)
    pred_parser.add_argument("--diameter", type=float, default=0.05)
    pred_parser.add_argument("--temp", type=float, default=35.0)
    pred_parser.add_argument("--o2", type=float, default=6.0)
    pred_parser.add_argument("--confidence", type=float, default=0.90)

    args = parser.parse_args()

    if args.command == "flow":
        res = calculate_flow_corrosion_kinetics(
            fluid_velocity_m_s=args.velocity,
            pipe_diameter_m=args.diameter,
            temp_C=args.temp,
            dissolved_o2_ppm=args.o2
        )
        print("\n--- Flow-Accelerated Corrosion Analysis ---")
        print(f"Reynolds Number:  {res.reynolds_number} ({res.flow_regime})")
        print(f"Schmidt Number:   {res.schmidt_number}")
        print(f"Sherwood Number:  {res.sherwood_number}")
        print(f"Limiting Current: {res.limiting_current_density_A_m2} A/m^2")
        print(f"Corrosion Rate:   {res.flow_accelerated_corrosion_mmyr} mm/year\n")

    elif args.command == "pitting":
        res = evaluate_pitting_susceptibility(
            cr_pct=args.cr,
            mo_pct=args.mo,
            n_pct=args.n,
            chloride_ppm=args.chloride,
            temp_C=args.temp
        )
        print("\n--- Localized Pitting Attack Analysis ---")
        print(f"PREN:             {res.pren}")
        print(f"Alloy Class:      {res.alloy_class}")
        print(f"Effective CPT:    {res.effective_cpt_celsius} °C")
        print(f"Operating Temp:   {res.temperature_C} °C")
        print(f"Pitting Risk:     {'CRITICAL RISK' if res.is_pitting_risk else 'PASSIVE / SAFE'}\n")

    elif args.command == "predict":
        flow = calculate_flow_corrosion_kinetics(
            fluid_velocity_m_s=args.velocity,
            pipe_diameter_m=args.diameter,
            temp_C=args.temp,
            dissolved_o2_ppm=args.o2
        )
        uq = ConformalCorrosionUQ().predict_interval(
            flow.flow_accelerated_corrosion_mmyr,
            confidence=args.confidence
        )
        print("\n--- CorroSense Conformal Corrosion Forecast ---")
        print(f"Predicted Rate:   {uq['predicted_mmyr']} mm/year")
        print(f"Confidence Level: {int(uq['confidence_level'] * 100)}%")
        print(f"Conformal Bounds: [{uq['lower_bound_mmyr']}, {uq['upper_bound_mmyr']}] mm/year")
        print(f"Uncertainty Margin: +/- {uq['q_critical_mmyr']} mm/year\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
