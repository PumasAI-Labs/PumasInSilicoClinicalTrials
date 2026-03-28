# PumasVpopWorkflow

Julia/Pumas implementation of the In Silico Clinical Trial (ISCT) workflow from the paper:

> **"A Step-by-Step Workflow for Performing In Silico Clinical Trials With Nonlinear Mixed Effects
> Models"** Cortés-Ríos et al., CPT: Pharmacometrics & Systems Pharmacology (2025) DOI:
> 10.1002/psp4.70122

## Workflow Overview

The package implements the six-step ISCT workflow from the paper:

1. **Model definition** — Define the pharmacometric or QSP model in Pumas
2. **Sensitivity analysis** — Identify influential parameters via Sobol and eFAST methods
3. **Structural identifiability** — Assess whether model parameters are identifiable from outputs
4. **Virtual population generation** — Sample plausible parameter sets using Gaussian copulas
5. **MILP calibration** — Calibrate the virtual population to clinical distributions using mixed
   integer linear programming
6. **Virtual clinical trial simulation** —
   Run multi-arm VCT simulations with the calibrated population

## Case Studies

### Tumor Burden Model

A 3-parameter tumor burden model used
as an introductory example to demonstrate each workflow step with minimal complexity.

### HBV QSP Model

An 11-ODE hepatitis B virus QSP model
that demonstrates the full workflow on a complex mechanistic system with multiple biomarkers
and treatment arms.

## Tutorials

### Tumor Burden Series

| # | Tutorial | Description |
|---|---------|-------------|
| 1 | [Model Introduction](tutorials/tb_01_model_introduction_tutorial.qmd) | Define and simulate the tumor burden model |
| 2 | [Global Sensitivity Analysis](tutorials/tb_02_gsa_tutorial.qmd) | Sobol and eFAST sensitivity analysis |
| 3 | [Structural Identifiability](tutorials/tb_03_structural_identifiability_tutorial.qmd) | Parameter identifiability assessment |
| 4 | [Copula Virtual Population](tutorials/tb_04_copula_vpop_tutorial.qmd) | Generate virtual patients via Gaussian copulas |
| 5 | [MILP Calibration](tutorials/tb_05_milp_calibration_tutorial.qmd) | Calibrate Vpop to clinical distributions |
| 6 | [VCT Simulation](tutorials/tb_06_vct_simulation_tutorial.qmd) | Run virtual clinical trials |

### HBV Series

| # | Tutorial | Description |
|---|---------|-------------|
| 1 | [Model Introduction](tutorials/hbv_01_model_introduction_tutorial.qmd) | Define and simulate the HBV QSP model |
| 2 | [Global Sensitivity Analysis](tutorials/hbv_02_gsa_tutorial.qmd) | Sensitivity analysis for 11-ODE system |
| 3 | [Structural Identifiability](tutorials/hbv_03_structural_identifiability_tutorial.qmd) | Identifiability with multiple outputs |
| 4 | [Copula Virtual Population](tutorials/hbv_04_copula_vpop_tutorial.qmd) | Generate virtual HBV patients |
| 5 | [MILP Calibration](tutorials/hbv_05_milp_calibration_tutorial.qmd) | Multi-variable Vpop calibration |
| 6 | [VCT Simulation](tutorials/hbv_06_vct_simulation_tutorial.qmd) | Multi-arm treatment comparison |

## Getting Started

### Requirements

- **Pumas 2.8.0** — see `Project.toml` for the complete dependency list

### Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Preview the documentation site

```bash
quarto preview
```

## Module Components

- **Models** — Tumor burden (3-parameter) and HBV QSP (11-ODE) Pumas models
- **Sensitivity** — Global sensitivity analysis (Sobol, eFAST)
- **Identifiability** - Structural identifiability
- **Sampling** — Gaussian copula-based virtual population generation
- **Calibration** — MILP-based Vpop calibration to clinical distributions
- **Simulation** — VCT simulation framework with multi-arm support
- **Visualization** — AlgebraOfGraphics + CairoMakie plotting utilities
