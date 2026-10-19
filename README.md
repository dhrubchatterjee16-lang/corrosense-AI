# CorroSense AI: Electrochemical, Nernst-Planck & Flow Corrosion Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CorroSenseAI-yellow)](https://huggingface.co/spaces/dhrubchatterjee/CorroSenseAI)
[![Tests: Passing](https://img.shields.io/badge/tests-12%2F12%20passed-brightgreen.svg)](tests/)

**CorroSense AI** is a first-principles electrochemical engineering and Scientific Machine Learning (SciML) platform for predicting flow-accelerated corrosion, localized pitting attack, and passive film electrodiffusion across infrastructure alloys.

---

## 🔬 Scientific Foundations & Governing Equations

CorroSense AI bridges hydrodynamic boundary layers with electrochemical charge transfer and passive film electrodiffusion:

### 1. Flow-Accelerated Corrosion (FAC) & Hydrodynamics
Fluid mass transfer is governed by the turbulent Berger-Hau / Dittus-Boelter Sherwood correlation:
$$\mathrm{Re} = \frac{u \cdot d}{\nu}, \quad \mathrm{Sc} = \frac{\nu}{D_{\mathrm{eff}}}, \quad \mathrm{Sh} = 0.023 \, \mathrm{Re}^{0.8} \, \mathrm{Sc}^{0.33}$$

Mass-transfer limited cathodic reduction current density ($i_L$) and Faraday corrosion penetration rate:
$$i_L = n F k_m C_{\mathrm{bulk}}, \quad \mathrm{CR}_{\mathrm{FAC}} = \frac{i_L \cdot M}{z F \rho} \times 1000 \times 3.1536 \times 10^7 \, (\mathrm{mm/year})$$

### 2. Nernst-Planck Electrodiffusion Flux
Across passive oxide films and boundary layers, ion migration and diffusion are conserved:
$$J_i = - D_i \nabla c_i - \frac{z_i D_i F}{R T} c_i \nabla \phi$$
$$i_{\mathrm{film}} = \sum_i z_i F J_i$$

### 3. Localized Pitting Attack & ASTM G48 CPT
Pitting Resistance Equivalent Number (PREN) and Critical Pitting Temperature ($CPT$):
$$\mathrm{PREN} = \% \mathrm{Cr} + 3.3 \% \mathrm{Mo} + 16 \% \mathrm{N}$$
$$\mathrm{CPT}_{\mathrm{effective}} = 2.5 \cdot \mathrm{PREN} - 31.0 - 4.0 \cdot \log_{10}\left(\frac{[\mathrm{Cl}^-]}{1000}\right)$$

---

## ⚡ Quickstart

### Installation

```bash
pip install -e .
```

### Python API

```python
from corrosense import (
    calculate_flow_corrosion_kinetics,
    evaluate_pitting_susceptibility,
    simulate_nernst_planck_film,
    ConformalCorrosionUQ
)

# 1. Flow-Accelerated Corrosion in Turbulent Pipe Flow
flow = calculate_flow_corrosion_kinetics(
    fluid_velocity_m_s=2.5,
    pipe_diameter_m=0.05,
    temp_C=35.0,
    dissolved_o2_ppm=6.5
)
print(f"Reynolds Number: {flow.reynolds_number} ({flow.flow_regime})")
print(f"Limiting Current Density: {flow.limiting_current_density_A_m2:.4f} A/m^2")
print(f"Corrosion Rate: {flow.flow_accelerated_corrosion_mmyr:.3f} mm/year")

# 2. Conformal Uncertainty Quantification (90% Finite-Sample Coverage)
uq = ConformalCorrosionUQ().predict_interval(flow.flow_accelerated_corrosion_mmyr, confidence=0.90)
print(f"90% Conformal Interval: [{uq['lower_bound_mmyr']}, {uq['upper_bound_mmyr']}] mm/year")

# 3. Localized Pitting Attack for Stainless Alloys (e.g., 2507 Super Duplex)
pitting = evaluate_pitting_susceptibility(
    cr_pct=25.0,
    mo_pct=4.0,
    n_pct=0.28,
    chloride_ppm=19000.0,
    temp_C=45.0
)
print(f"PREN: {pitting.pren} -> Class: {pitting.alloy_class}")
print(f"Effective CPT: {pitting.effective_cpt_celsius:.1f} °C (Pitting Risk: {pitting.is_pitting_risk})")
```

---

## 🖥️ Command-Line Interface (CLI)

```bash
# Calculate Flow Corrosion
corrosense flow --velocity 2.0 --diameter 0.05 --temp 30.0 --o2 7.0

# Evaluate Pitting Susceptibility & PREN
corrosense pitting --cr 18.0 --mo 2.5 --n 0.15 --chloride 19000 --temp 40.0

# Conformal Uncertainty Forecast
corrosense predict --velocity 2.5 --diameter 0.05 --temp 40.0 --confidence 0.95
```

---

## 📊 Streamlit Interactive Dashboard

Launch the full-stack visual analytics dashboard:
```bash
streamlit run corrosion_app_FIXED_1.py
```
Features:
- Real-time severity prediction over 5,000 empirical corrosion records across 63 material families.
- Permutation feature importance and live training convergence plots.
- Alloy material selection and corrosion mitigation advisor.


---

## 🧪 Automated Verification Suite

Run the full electrochemical and hydrodynamic test suite:
```bash
python run_tests.py
```
```text
Ran 12 tests in 0.003s
OK - ALL CORROSENSE-AI TESTS PASSED (12/12)!
```

---

## 📚 References
- Fontana, M. G. (1986). *Corrosion Engineering*. McGraw-Hill.
- ASTM G102-89: *Standard Practice for Calculation of Corrosion Rates and Related Information from Electrochemical Measurements*.
- ASTM G48-11: *Standard Test Methods for Pitting and Crevice Corrosion Resistance of Stainless Steels*.
- Berger, F. P., & Hau, K. F-F. L. (1977). *Mass transfer in turbulent pipe flow*. International Journal of Heat and Mass Transfer, 20(11), 1185-1194.
- Newman, J., & Thomas-Alyea, K. E. (2004). *Electrochemical Systems*. John Wiley & Sons.

---

## 📄 License
MIT License. Copyright (c) 2026 Dhrub Chatterjee.
