"""
    HBV Parameter Sampling

This module implements parameter sampling for the HBV QSP model virtual populations.

The HBV model has:
- 22 fixed parameters (same for all virtual patients)
- 9 estimated parameters with inter-individual variability:
  1. beta - Infection rate (log10)
  2. p_S - HBsAg production rate (log10)
  3. m - Immune killing rate (log10)
  4. k_Z - Z production rate (log10)
  5. convE - E conversion factor
  6. epsNUC - NUC efficacy (log10)
  7. epsIFN - IFN efficacy (log10)
  8. r_E_IFN - IFN effect on E cells
  9. k_D - D production rate (log10)

For virtual population generation, parameters can be:
1. Loaded directly from CSV files (pre-sampled from NLME)
2. Generated using Gaussian copulas (if distribution parameters known)
"""

using CSV
using DataFrames
using DataFramesMeta
using Statistics
using Copulas
using Pumas.Distributions
using Random
using LinearAlgebra

#=============================================================================
# HBV Parameter Names and Indices
=============================================================================#

"""
Names of the 9 estimated parameters with IIV.
"""
const HBV_ESTIMATED_PARAMS = [:beta, :p_S, :m, :k_Z, :convE, :epsNUC, :epsIFN, :r_E_IFN, :k_D]

"""
Names of the 22 fixed parameters.
"""
const HBV_FIXED_PARAM_NAMES = [
    :iniV, :p_V, :r_T, :r_E, :T_max, :n, :phiE, :dEtoX, :phiQ,
    :d_V, :d_TI, :d_E, :d_Z, :d_Q, :rho, :r_X, :d_X, :d_Y,
    :Smax, :phiS, :nS, :d_D,
]

"""
Default fixed parameter values for HBV model.
"""
const HBV_FIXED_VALUES = Dict(
    :iniV => -0.481486,
    :p_V => 2.0,
    :r_T => 1.0,
    :r_E => 0.1,
    :T_max => 13600000.0,
    :n => 2.0,
    :phiE => 2.0,
    :dEtoX => 0.3,
    :phiQ => 0.8,
    :d_V => 0.67,
    :d_TI => 0.0039,
    :d_E => -2.0,
    :d_Z => -0.328,
    :d_Q => -2.414,
    :rho => -3.0,
    :r_X => 1.0,
    :d_X => 0.2,
    :d_Y => 0.22,
    :Smax => 1.0,
    :phiS => 0.147,
    :nS => 0.486,
    :d_D => -0.62157,
    :iniZ => 1.25
)

#=============================================================================
# CSV Loading Functions
=============================================================================#

"""
    load_hbv_vpop(filepath::String) -> DataFrame

Load a pre-sampled HBV virtual population from a CSV file.

The CSV should contain columns for all 31 parameters plus an id column.

# Example
```julia
vpop = load_hbv_vpop("sampledvps_100000_01.csv")
```
"""
function load_hbv_vpop(filepath::String)
    df = CSV.read(filepath, DataFrame)

    # Ensure id column exists
    if !hasproperty(df, :id)
        df.id = 1:nrow(df)
    end

    return df
end

"""
    extract_estimated_params(vpop::DataFrame) -> DataFrame

Extract only the 9 estimated parameters from a full HBV Vpop DataFrame.
"""
function extract_estimated_params(vpop::DataFrame)
    cols = [:id, HBV_ESTIMATED_PARAMS...]
    return vpop[:, cols]
end

"""
    add_fixed_params(estimated_df::DataFrame) -> DataFrame

Add fixed parameter columns to a DataFrame containing only estimated parameters.
"""
function add_fixed_params(estimated_df::DataFrame)
    df = copy(estimated_df)

    for (param, value) in HBV_FIXED_VALUES
        if !hasproperty(df, param)
            df[!, param] .= value
        end
    end

    return df
end

#=============================================================================
# Parameter Distribution Estimation
=============================================================================#

"""
    HBVParameterStats

Statistics for HBV estimated parameters computed from a virtual population.
"""
struct HBVParameterStats
    means::Dict{Symbol, Float64}
    stds::Dict{Symbol, Float64}
    correlation::Matrix{Float64}
    param_names::Vector{Symbol}
end

"""
    compute_hbv_stats(vpop::DataFrame) -> HBVParameterStats

Compute mean, standard deviation, and correlation matrix for HBV estimated parameters.
"""
function compute_hbv_stats(vpop::DataFrame)
    param_names = HBV_ESTIMATED_PARAMS
    n_params = length(param_names)

    # Extract parameter matrix
    param_matrix = Matrix(vpop[:, param_names])

    # Compute statistics
    means = Dict(name => mean(vpop[!, name]) for name in param_names)
    stds = Dict(name => std(vpop[!, name]) for name in param_names)

    # Compute correlation matrix
    correlation = cor(param_matrix)

    return HBVParameterStats(means, stds, correlation, collect(param_names))
end

"""
    print_hbv_stats(stats::HBVParameterStats)

Print formatted statistics for HBV parameters.
"""
function print_hbv_stats(stats::HBVParameterStats)
    println("HBV Parameter Statistics")
    println("="^50)
    println("\nParameter Means and Standard Deviations:")
    for name in stats.param_names
        println("  $(name): μ = $(round(stats.means[name], digits = 4)), σ = $(round(stats.stds[name], digits = 4))")
    end

    println("\nCorrelation Matrix:")
    for (i, name) in enumerate(stats.param_names)
        row = [round(stats.correlation[i, j], digits = 2) for j in 1:length(stats.param_names)]
        println("  $(name): $row")
    end
    return
end

#=============================================================================
# Copula-Based Sampling for HBV
=============================================================================#

"""
    HBVParameterSpec

Specification for an HBV estimated parameter.
All parameters are assumed to follow normal distributions on the specified scale.
"""
struct HBVParameterSpec
    name::Symbol
    mean::Float64
    std::Float64
end

"""
    create_hbv_param_specs(stats::HBVParameterStats) -> Vector{HBVParameterSpec}

Create parameter specifications from computed statistics.
"""
function create_hbv_param_specs(stats::HBVParameterStats)
    return [
        HBVParameterSpec(name, stats.means[name], stats.stds[name])
            for name in stats.param_names
    ]
end

"""
    generate_hbv_vpop(
        n::Int,
        param_specs::Vector{HBVParameterSpec},
        correlation_matrix::Matrix{Float64};
        seed::Union{Int,Nothing} = nothing,
        include_fixed::Bool = true
    ) -> DataFrame

Generate a virtual population for HBV using Gaussian copulas.

# Arguments
- `n`: Number of virtual patients to generate
- `param_specs`: Vector of HBVParameterSpec for estimated parameters
- `correlation_matrix`: Correlation matrix for estimated parameters
- `seed`: Random seed for reproducibility
- `include_fixed`: Whether to include fixed parameters in output

# Returns
DataFrame with columns for each parameter and `id` column
"""
function generate_hbv_vpop(
        n::Int,
        param_specs::Vector{HBVParameterSpec},
        correlation_matrix::Matrix{Float64};
        seed::Union{Int, Nothing} = nothing,
        include_fixed::Bool = true
    )
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_params = length(param_specs)
    @assert size(correlation_matrix) == (n_params, n_params) "Correlation matrix size mismatch"

    # Create marginal distributions
    marginals = [Normal(spec.mean, spec.std) for spec in param_specs]

    # Create Gaussian copula
    copula = GaussianCopula(correlation_matrix)

    # Generate correlated uniform samples
    U = rand(copula, n)'

    # Transform to target marginals
    Y = similar(U)
    for (j, marginal) in enumerate(marginals)
        Y[:, j] = quantile.(marginal, U[:, j])
    end

    # Create DataFrame
    df = DataFrame(Y, [spec.name for spec in param_specs])
    df.id = 1:n

    # Add fixed parameters if requested
    if include_fixed
        df = add_fixed_params(df)
    end

    # Reorder columns to put id first
    select!(df, :id, Not(:id))

    return df
end

"""
    generate_hbv_vpop_from_csv(
        n::Int,
        reference_csv::String;
        seed::Union{Int,Nothing} = nothing,
        include_fixed::Bool = true
    ) -> DataFrame

Generate a new HBV virtual population using statistics computed from a reference CSV.

This function:
1. Loads the reference virtual population
2. Computes mean, std, and correlation for estimated parameters
3. Generates new samples using Gaussian copulas

# Example
```julia
vpop = generate_hbv_vpop_from_csv(10000, "Parameters_Naive.csv"; seed=42)
```
"""
function generate_hbv_vpop_from_csv(
        n::Int,
        reference_csv::String;
        seed::Union{Int, Nothing} = nothing,
        include_fixed::Bool = true
    )
    # Load reference population
    ref_vpop = load_hbv_vpop(reference_csv)

    # Compute statistics
    stats = compute_hbv_stats(ref_vpop)

    # Create parameter specs
    param_specs = create_hbv_param_specs(stats)

    # Generate new population
    return generate_hbv_vpop(
        n,
        param_specs,
        stats.correlation;
        seed = seed,
        include_fixed = include_fixed
    )
end

"""
    subsample_hbv_vpop(
        vpop::DataFrame,
        n::Int;
        seed::Union{Int,Nothing} = nothing
    ) -> DataFrame

Randomly subsample n virtual patients from an existing Vpop.
"""
function subsample_hbv_vpop(
        vpop::DataFrame,
        n::Int;
        seed::Union{Int, Nothing} = nothing
    )
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_available = nrow(vpop)
    @assert n <= n_available "Cannot subsample more patients than available"

    indices = randperm(n_available)[1:n]
    subsampled = vpop[indices, :]

    # Reset IDs
    subsampled.id = 1:n

    return subsampled
end

#=============================================================================
# Summary Functions
=============================================================================#

"""
    summarize_hbv_vpop(vpop::DataFrame) -> DataFrame

Generate summary statistics for an HBV virtual population.
"""
function summarize_hbv_vpop(vpop::DataFrame)
    param_names = HBV_ESTIMATED_PARAMS

    summaries = DataFrame(
        parameter = Symbol[],
        n = Int[],
        mean = Float64[],
        median = Float64[],
        std = Float64[],
        min = Float64[],
        max = Float64[],
        q05 = Float64[],
        q25 = Float64[],
        q75 = Float64[],
        q95 = Float64[]
    )

    for name in param_names
        if hasproperty(vpop, name)
            values = vpop[!, name]
            push!(
                summaries, (
                    parameter = name,
                    n = length(values),
                    mean = mean(values),
                    median = median(values),
                    std = std(values),
                    min = minimum(values),
                    max = maximum(values),
                    q05 = quantile(values, 0.05),
                    q25 = quantile(values, 0.25),
                    q75 = quantile(values, 0.75),
                    q95 = quantile(values, 0.95),
                )
            )
        end
    end

    return summaries
end

export HBV_ESTIMATED_PARAMS, HBV_FIXED_PARAM_NAMES, HBV_FIXED_VALUES
export load_hbv_vpop, extract_estimated_params, add_fixed_params
export HBVParameterStats, compute_hbv_stats, print_hbv_stats
export HBVParameterSpec, create_hbv_param_specs
export generate_hbv_vpop, generate_hbv_vpop_from_csv, subsample_hbv_vpop
export summarize_hbv_vpop
