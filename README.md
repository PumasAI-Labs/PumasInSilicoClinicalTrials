# PumasVpopWorkflow

Julia/Pumas implementation of the In Silico Clinical Trial (ISCT) workflow from the paper:

> **"A Step-by-Step Workflow for Performing In Silico Clinical Trials With Nonlinear Mixed Effects Models"**
> Cortés-Ríos et al., CPT: Pharmacometrics & Systems Pharmacology (2025)
> DOI: 10.1002/psp4.70122

## Status

**Note:** The MILP calibration module is not yet complete.

## Getting Started

This project is **not** designed to be used as a standard Julia package. Instead, follow the step-by-step instructions below.

### Step 1: Open Julia at the Project Root

Navigate to the repository root and start Julia with the project environment:

```bash
cd /path/to/PumasVpopWorkflow
julia --project=.
```

### Step 2: Install Dependencies

On first use, install all required dependencies:

```julia
using Pkg
Pkg.instantiate()
```

### Step 3: Load the Module

Load the ISCTWorkflow module by including the source file:

```julia
include("src/ISCTWorkflow.jl")
```

This will define the `ISCTWorkflow` module with all exported functions.

### Step 4: Run Examples

Open any example file from the `examples/` folder and run the code step by step. Each example uses `using .ISCTWorkflow` (with the leading dot) to import the module from the current namespace.

Available examples:

| Example | Description |
|---------|-------------|
| `01_tumor_burden_isct.jl` | Complete ISCT workflow for the tumor burden model |
| `02_hbv_vpop_analysis.jl` | HBV virtual population analysis and parameter sampling |
| `03_milp_calibration.jl` | MILP-based virtual population calibration (incomplete) |
| `04_vct_simulation.jl` | Virtual clinical trial simulation framework |
| `05_gsa_analysis.jl` | Global sensitivity analysis with Sobol/eFAST |
| `06_visualization.jl` | Comprehensive visualization examples |
| `07_hbv_dynamics_visualization.jl` | HBV population dynamics visualization |

## Module Components

- **Models**: Tumor burden (3-parameter) and HBV QSP (11-ODE) Pumas models
- **Sampling**: Gaussian copula-based virtual population generation
- **Calibration**: MILP-based Vpop calibration to clinical distributions (incomplete)
- **Simulation**: VCT simulation framework with multi-arm support
- **Sensitivity**: Global sensitivity analysis (Sobol, eFAST)
- **Visualization**: AlgebraOfGraphics + CairoMakie plotting utilities

## Requirements

- Julia 1.10+
- Pumas
- JuMP + HiGHS (for MILP calibration)
- Copulas.jl
- GlobalSensitivity.jl
- AlgebraOfGraphics + CairoMakie (for visualization)

See `Project.toml` for the complete list of dependencies.

## Output

Generated plots are saved to the `outputs/` directory. Create it if needed:

```julia
mkpath("outputs")
```
