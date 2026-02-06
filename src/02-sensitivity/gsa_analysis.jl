#=
Global Sensitivity Analysis Module

Implements Step 7 of the ISCT workflow: "Analyze VCT results with GSA"

This module provides utilities for running global sensitivity analysis
on the tumor burden and HBV models using Pumas.gsa() with Sobol and eFAST methods.

Reference:
    Section 2, Step 7 of the ISCT workflow paper
=#

using Pumas
using GlobalSensitivity
using DataFrames
using Statistics
using LinearAlgebra: Diagonal
using Printf

#=============================================================================
# GSA Parameter Specification
=============================================================================#

"""
    GSAParameterRange

Specifies the range for a parameter in GSA analysis.

# Fields
- `name::Symbol`: Parameter name
- `lower::Float64`: Lower bound
- `upper::Float64`: Upper bound
- `scale::Symbol`: Parameter scale (:linear or :log)
"""
struct GSAParameterRange
    name::Symbol
    lower::Float64
    upper::Float64
    scale::Symbol

    function GSAParameterRange(name::Symbol, lower::Float64, upper::Float64; scale::Symbol=:linear)
        @assert lower < upper "Lower bound must be less than upper bound"
        @assert scale in (:linear, :log) "Scale must be :linear or :log"
        new(name, lower, upper, scale)
    end
end

"""
    GSAResult

Container for GSA analysis results.

# Fields
- `first_order::DataFrame`: First-order Sobol indices
- `total_order::DataFrame`: Total-order Sobol indices
- `parameters::Vector{Symbol}`: Parameter names
- `outputs::Vector{Symbol}`: Output variable names
- `method::Symbol`: GSA method used (:sobol or :efast)
- `n_samples::Int`: Number of samples used
"""
struct GSAResult
    first_order::DataFrame
    total_order::DataFrame
    parameters::Vector{Symbol}
    outputs::Vector{Symbol}
    method::Symbol
    n_samples::Int
end

# Note: Duplicate GSA models have been removed.
# GSA now uses the main models (tumor_burden_model, hbv_model) with constantcoef
# to fix random effect variances. The models include @observed blocks for GSA endpoints.

#=============================================================================
# Default Parameter Ranges
=============================================================================#

"""
    TUMOR_BURDEN_GSA_PARAMS

Default parameter ranges for tumor burden model GSA.
Based on the distributions from Qi & Cao (2023).
"""
const TUMOR_BURDEN_GSA_PARAMS = [
    GSAParameterRange(:f, 0.05, 0.95),      # Treatment-sensitive fraction
    GSAParameterRange(:g, 0.0005, 0.005),    # Growth rate (d⁻¹)
    GSAParameterRange(:k, 0.002, 0.05),      # Death rate (d⁻¹)
]

"""
    HBV_GSA_PARAMS

Default parameter ranges for HBV model GSA.
Based on the estimated parameter distributions.
"""
const HBV_GSA_PARAMS = [
    GSAParameterRange(:beta, -8.0, -2.0),        # Infection rate (log10)
    GSAParameterRange(:p_S, 6.0, 10.0),          # HBsAg production (log10)
    GSAParameterRange(:m, 1.0, 4.0),             # Immune killing (log10)
    GSAParameterRange(:k_Z, -7.0, -3.0),         # ALT response (log10)
    GSAParameterRange(:convE, 1.0, 4.0),         # Effector conversion (log10)
    GSAParameterRange(:epsNUC, -4.0, 0.0),       # NUC efficacy (log10)
    GSAParameterRange(:epsIFN, -3.0, 0.0),       # IFN efficacy (log10)
    GSAParameterRange(:r_E_IFN, 0.1, 0.5),       # IFN immune boost
    GSAParameterRange(:k_D, -6.0, -2.0),         # DC activation (log10)
]

#=============================================================================
# GSA Helper Functions
=============================================================================#

"""
    create_param_ranges(params::Vector{GSAParameterRange})

Convert GSAParameterRange vector to low/high NamedTuples for Pumas.gsa().

# Returns
Tuple of (p_range_low, p_range_high) NamedTuples
"""
function create_param_ranges(params::Vector{GSAParameterRange})
    names = [p.name for p in params]
    lows = [p.lower for p in params]
    highs = [p.upper for p in params]

    p_range_low = NamedTuple{Tuple(names)}(lows)
    p_range_high = NamedTuple{Tuple(names)}(highs)

    return (p_range_low, p_range_high)
end

"""
    create_constant_coef(model_params::NamedTuple, vary_params::Vector{Symbol})

Create constantcoef NamedTuple for parameters NOT being varied in GSA.

# Arguments
- `model_params`: All model parameters
- `vary_params`: Parameters being varied (to exclude)

# Returns
NamedTuple of constant parameters
"""
function create_constant_coef(model_params::NamedTuple, vary_params::Vector{Symbol})
    const_names = Symbol[]
    const_values = Any[]

    for (name, value) in pairs(model_params)
        if !(name in vary_params)
            push!(const_names, name)
            push!(const_values, value)
        end
    end

    return NamedTuple{Tuple(const_names)}(const_values)
end

#=============================================================================
# Tumor Burden GSA
=============================================================================#

"""
    run_tumor_burden_gsa(;
        method::Symbol = :sobol,
        n_samples::Int = 1000,
        observation_times = 0:7:126,
        treatment::Int = 1,
        param_ranges::Vector{GSAParameterRange} = TUMOR_BURDEN_GSA_PARAMS,
        outputs::Vector{Symbol} = [:final_tumor, :auc_tumor]
    ) -> GSAResult

Run global sensitivity analysis on the tumor burden model.

# Arguments
- `method`: GSA method (:sobol or :efast)
- `n_samples`: Number of samples (N for Sobol, n for eFAST)
- `observation_times`: Simulation time points
- `treatment`: Treatment arm (0=control, 1=treatment)
- `param_ranges`: Parameter ranges for sensitivity analysis
- `outputs`: Output variables to analyze

# Returns
GSAResult with first-order and total-order indices

# Example
```julia
result = run_tumor_burden_gsa(method=:efast, n_samples=500)
println(result.first_order)
```
"""
function run_tumor_burden_gsa(;
    method::Symbol = :sobol,
    n_samples::Int = 1000,
    observation_times = 0:7:126,
    treatment::Int = 1,
    param_ranges::Vector{GSAParameterRange} = TUMOR_BURDEN_GSA_PARAMS,
    outputs::Vector{Symbol} = [:final_tumor, :auc_tumor]
)
    # Create subject
    subject = Subject(
        id = 1,
        covariates = (treatment = treatment,),
        time = collect(observation_times)
    )

    # Base parameters - uses tumor_burden_model (with random effects)
    # We fix Ω and σ to tiny values via constantcoef to make simulation deterministic
    base_params = (
        tvf = 0.27,
        tvg = 0.0013,
        tvk = 0.0091,
        Ω = Diagonal([1e-10, 1e-10, 1e-10]),  # Tiny random effect variance
        σ = 1e-10                              # Tiny residual error
    )

    # Create parameter ranges for typical values (GSA varies these)
    # Map from individual params (f, g, k) to population params (tvf, tvg, tvk)
    p_low = (tvf = param_ranges[1].lower, tvg = param_ranges[2].lower, tvk = param_ranges[3].lower)
    p_high = (tvf = param_ranges[1].upper, tvg = param_ranges[2].upper, tvk = param_ranges[3].upper)
    vary_names = [:tvf, :tvg, :tvk]

    # Fix Ω and σ to prevent them from being varied
    # const_coef = (Ω = Diagonal([1e-10, 1e-10, 1e-10]), σ = 1e-10)
    const_coef = (:Ω, :σ)

    # Run GSA using the main model
    gsa_method = method == :sobol ? GlobalSensitivity.Sobol() : GlobalSensitivity.eFAST()

    gsa_result = if method == :sobol
        Pumas.gsa(
            tumor_burden_model,
            subject,
            base_params,
            gsa_method,
            outputs,
            p_low,
            p_high;
            constantcoef = const_coef,
            N = n_samples
        )
    else
        Pumas.gsa(
            tumor_burden_model,
            subject,
            base_params,
            gsa_method,
            outputs,
            p_low,
            p_high;
            constantcoef = const_coef,
            n = n_samples
        )
    end

    # Convert to DataFrame - use original param names for output
    output_names = [:f, :g, :k]  # Map back to user-friendly names
    first_order_df = _gsa_to_dataframe(gsa_result.first_order, vary_names, outputs, output_names)
    total_order_df = _gsa_to_dataframe(gsa_result.total_order, vary_names, outputs, output_names)

    return GSAResult(
        first_order_df,
        total_order_df,
        output_names,
        outputs,
        method,
        n_samples
    )
end

"""
    _gsa_to_dataframe(gsa_matrix, param_names, output_names, display_names=nothing)

Convert GSA result matrix to tidy DataFrame format.

# Arguments
- `gsa_result`: GSA result from Pumas.gsa()
- `param_names`: Parameter names as used in the model
- `output_names`: Output variable names
- `display_names`: Optional alternative names for display (same order as param_names)
"""
function _gsa_to_dataframe(
    gsa_result,
    param_names::Vector{Symbol},
    output_names::Vector{Symbol},
    display_names::Union{Vector{Symbol}, Nothing} = nothing
)
    rows = NamedTuple[]
    names_to_use = isnothing(display_names) ? param_names : display_names

    for (i, output) in enumerate(output_names)
        for (j, param) in enumerate(param_names)
            idx = gsa_result[i, :]
            value = haskey(idx, param) ? idx[param] : NaN
            push!(rows, (
                output = output,
                parameter = names_to_use[j],
                index = value
            ))
        end
    end

    return DataFrame(rows)
end

#=============================================================================
# HBV GSA
=============================================================================#

"""
    run_hbv_gsa(;
        method::Symbol = :efast,
        n_samples::Int = 500,
        treatment::Symbol = :NUC_IFN,
        observation_times = 0:7:336,
        param_ranges::Vector{GSAParameterRange} = HBV_GSA_PARAMS,
        outputs::Vector{Symbol} = [:final_hbsag, :final_viral, :hbsag_nadir]
    ) -> GSAResult

Run global sensitivity analysis on the HBV model.

# Arguments
- `method`: GSA method (:sobol or :efast). Default :efast (faster)
- `n_samples`: Number of samples
- `treatment`: Treatment type (:NUC, :IFN, or :NUC_IFN)
- `observation_times`: Simulation time points
- `param_ranges`: Parameter ranges for sensitivity analysis
- `outputs`: Output variables to analyze

# Returns
GSAResult with first-order and total-order indices

# Example
```julia
result = run_hbv_gsa(method=:efast, n_samples=500, treatment=:NUC_IFN)
println(result.first_order)
```
"""
function run_hbv_gsa(;
    method::Symbol = :efast,
    n_samples::Int = 500,
    treatment::Symbol = :NUC_IFN,
    observation_times = 0:7:336,
    param_ranges::Vector{GSAParameterRange} = HBV_GSA_PARAMS,
    outputs::Vector{Symbol} = [:final_hbsag, :final_viral, :hbsag_nadir]
)
    # Set treatment covariates
    dNUC = treatment in (:NUC, :NUC_IFN) ? 1.0 : 0.0
    dIFN = treatment in (:IFN, :NUC_IFN) ? 1.0 : 0.0

    # Create subject
    subject = Subject(
        id = 1,
        covariates = (dNUC = dNUC, dIFN = dIFN),
        time = collect(observation_times)
    )

    # Base parameters (merge fixed + estimated)
    base_params = (
        # Fixed parameters
        iniV = -0.481486,
        p_V = 2.0,
        r_T = 1.0,
        r_E = 0.1,
        T_max = 13600000.0,
        n = 2.0,
        phiE = 2.0,
        dEtoX = 0.3,
        phiQ = 0.8,
        d_V = 0.67,
        d_TI = 0.0039,
        d_E = -2.0,
        d_Z = -0.328,
        d_Q = -2.414,
        rho = -3.0,
        r_X = 1.0,
        d_X = 0.2,
        d_Y = 0.22,
        Smax = 1.0,
        phiS = 0.147,
        nS = 0.486,
        d_D = -0.62157,
        iniZ = 1.25,
        # Estimated parameters (defaults)
        beta = -5.0,
        p_S = 8.0,
        m = 2.5,
        k_Z = -5.0,
        convE = 2.5,
        epsNUC = -2.0,
        epsIFN = -1.5,
        r_E_IFN = 0.3,
        k_D = -4.0
    )

    # Create parameter ranges
    p_low, p_high = create_param_ranges(param_ranges)
    vary_names = [p.name for p in param_ranges]

    # Create constant coefficients (fixed params)
    const_coef = create_constant_coef(base_params, vary_names)

    # Run GSA
    gsa_method = method == :sobol ? GlobalSensitivity.Sobol() : GlobalSensitivity.eFAST()

    gsa_result = if method == :sobol
        Pumas.gsa(
            hbv_model,  # Use main model with @observed block
            subject,
            base_params,
            gsa_method,
            outputs,
            p_low,
            p_high;
            constantcoef = const_coef,
            N = n_samples
        )
    else
        Pumas.gsa(
            hbv_model,  # Use main model with @observed block
            subject,
            base_params,
            gsa_method,
            outputs,
            p_low,
            p_high;
            constantcoef = const_coef,
            n = n_samples
        )
    end

    # Convert to DataFrame
    first_order_df = _gsa_to_dataframe(gsa_result.first_order, vary_names, outputs)
    total_order_df = _gsa_to_dataframe(gsa_result.total_order, vary_names, outputs)

    return GSAResult(
        first_order_df,
        total_order_df,
        vary_names,
        outputs,
        method,
        n_samples
    )
end

#=============================================================================
# GSA Result Analysis
=============================================================================#

"""
    summarize_gsa(result::GSAResult) -> DataFrame

Create summary table of GSA results with rankings.

# Returns
DataFrame with parameters ranked by importance for each output.
"""
function summarize_gsa(result::GSAResult)
    summary_rows = NamedTuple[]

    for output in result.outputs
        # Get first and total order for this output
        fo = filter(row -> row.output == output, result.first_order)
        to = filter(row -> row.output == output, result.total_order)

        # Sort by total order (most important first)
        sorted_to = sort(to, :index, rev=true)

        for (rank, row) in enumerate(eachrow(sorted_to))
            param = row.parameter
            fo_idx = filter(r -> r.parameter == param, fo)
            fo_val = nrow(fo_idx) > 0 ? fo_idx[1, :index] : NaN

            push!(summary_rows, (
                output = output,
                rank = rank,
                parameter = param,
                first_order = round(fo_val, digits=4),
                total_order = round(row.index, digits=4),
                interaction = round(row.index - fo_val, digits=4)
            ))
        end
    end

    return DataFrame(summary_rows)
end

"""
    get_influential_params(result::GSAResult; threshold::Float64=0.1) -> Vector{Symbol}

Identify parameters with total-order index above threshold.

# Arguments
- `result`: GSA result
- `threshold`: Minimum total-order index to be considered influential

# Returns
Vector of influential parameter names
"""
function get_influential_params(result::GSAResult; threshold::Float64=0.1)
    influential = Set{Symbol}()

    for row in eachrow(result.total_order)
        if row.index >= threshold
            push!(influential, row.parameter)
        end
    end

    return collect(influential)
end

"""
    print_gsa_summary(result::GSAResult)

Print formatted summary of GSA results.
"""
function print_gsa_summary(result::GSAResult)
    println("=" ^ 60)
    println("Global Sensitivity Analysis Results")
    println("=" ^ 60)
    println("Method: $(result.method)")
    println("Samples: $(result.n_samples)")
    println("Parameters: " * join(string.(result.parameters), ", "))
    println("Outputs: " * join(string.(result.outputs), ", "))
    println()

    summary = summarize_gsa(result)

    for output in result.outputs
        println("-" ^ 40)
        println("Output: $output")
        println("-" ^ 40)

        output_data = filter(row -> row.output == output, summary)

        println("Rank | Parameter     | First Order | Total Order | Interaction")
        println("-" ^ 60)

        for row in eachrow(output_data)
            println(@sprintf("  %d  | %-12s |   %6.4f    |   %6.4f    |   %6.4f",
                row.rank, row.parameter, row.first_order, row.total_order, row.interaction))
        end
        println()
    end

    # Highlight influential parameters
    influential = get_influential_params(result)
    if !isempty(influential)
        println("Most influential parameters (total order ≥ 0.1):")
        for p in influential
            println("  - $p")
        end
    end
end

#=============================================================================
# Exports
=============================================================================#

export GSAParameterRange, GSAResult
export TUMOR_BURDEN_GSA_PARAMS, HBV_GSA_PARAMS
# Note: tumor_burden_gsa_model and hbv_gsa_model removed - use main models with constantcoef
export run_tumor_burden_gsa, run_hbv_gsa
export create_param_ranges, create_constant_coef
export summarize_gsa, get_influential_params, print_gsa_summary
