"""
    Copula-based Parameter Sampling for Virtual Population Generation

This module implements Step 4 of the ISCT workflow (Figure 2 in paper):
"Generate Plausible Patients (Parameter Sampling)"

The method uses Gaussian copulas to generate correlated parameter samples
while maintaining NLME-estimated distributions and correlations.

Key concepts:
1. Use Gaussian copula to capture parameter correlations
2. Transform uniform marginals to target distributions (Normal, LogNormal, LogitNormal)
3. Preserve Spearman/Kendall rank correlations after transformation

Reference:
    Section 2, Step 4 of the ISCT workflow paper (Figure 2)
"""

using Copulas
using Pumas.Distributions
using Random
using DataFrames
using LinearAlgebra
using Statistics
using LogExpFunctions: logit, logistic
using StatsBase: tiedrank, corspearman

#=============================================================================
# Transformation Functions
#
# Note: logit and logistic functions are imported from LogExpFunctions.jl
# - logit(x) = log(x / (1-x))
# - logistic(y) = 1 / (1 + exp(-y))
=============================================================================#

"""
    bounded_logistic(y, a, b)

Inverse logit transformation to map from (-∞, +∞) to bounded range [a, b].

Formula: P = a + (b - a) * logistic(y)

Uses LogExpFunctions.logistic for numerical stability.
"""
bounded_logistic(y::Real, a::Real, b::Real) = a + (b - a) * logistic(y)

"""
    logit_transform_mean(P, a, b)

Calculate the logit-transformed mean (μ) for a parameter with median P
and bounds [a, b].

Formula: μ = logit((P - a) / (b - a))
"""
logit_transform_mean(P::Real, a::Real, b::Real) = logit((P - a) / (b - a))

#=============================================================================
# Parameter Distribution Specification
=============================================================================#

"""
    ParameterSpec

Specification for a single parameter's distribution in the virtual population.

# Fields
- `name::Symbol`: Parameter name
- `median::Float64`: Population median value
- `omega::Float64`: Inter-individual variability (SD on transformed scale)
- `lower::Float64`: Lower bound
- `upper::Float64`: Upper bound
- `transform::Symbol`: Distribution type (:logitnormal, :lognormal, :normal)
"""
struct ParameterSpec
    name::Symbol
    median::Float64
    omega::Float64
    lower::Float64
    upper::Float64
    transform::Symbol

    function ParameterSpec(name, median, omega, lower, upper, transform=:logitnormal)
        @assert lower < median < upper "Median must be within bounds"
        @assert omega > 0 "Omega must be positive"
        @assert transform in (:logitnormal, :lognormal, :normal) "Unknown transform"
        new(name, median, omega, lower, upper, transform)
    end
end

"""
    TumorBurdenParams()

Create parameter specifications for the Tumor Burden model (Section 3.1).

Parameters from Qi & Cao (2023):
- f: Treatment-sensitive fraction, median=0.27, ω=2.16, range [0,1]
- g: Growth rate, median=0.0013 d⁻¹, ω=1.57, range [0, 0.13]
- k: Death rate, median=0.0091 d⁻¹, ω=1.24, range [0, 1.6]
"""
function TumorBurdenParams()
    return [
        ParameterSpec(:f, 0.27, 2.16, 0.0, 1.0, :logitnormal),
        ParameterSpec(:g, 0.0013, 1.57, 0.0, 0.13, :logitnormal),
        ParameterSpec(:k, 0.0091, 1.24, 0.0, 1.6, :logitnormal)
    ]
end

"""
    TumorBurdenCorrelation()

Return the correlation matrix for the Tumor Burden model.

From paper: r(f,g) = -0.64, r(f,k) = 0, r(g,k) = 0
"""
function TumorBurdenCorrelation()
    r = -0.64
    return [
        1.0  r    0.0
        r    1.0  0.0
        0.0  0.0  1.0
    ]
end

#=============================================================================
# Core Sampling Functions
=============================================================================#

"""
    create_marginal_distribution(spec::ParameterSpec)

Create the marginal distribution on the transformed scale for sampling.

For logit-normal parameters:
- μ = logit((median - lower) / (upper - median))
- Returns Normal(μ, ω)
"""
function create_marginal_distribution(spec::ParameterSpec)
    if spec.transform == :logitnormal
        μ = logit_transform_mean(spec.median, spec.lower, spec.upper)
        return Normal(μ, spec.omega)
    elseif spec.transform == :lognormal
        μ = log(spec.median)
        return Normal(μ, spec.omega)
    else  # :normal
        return Normal(spec.median, spec.omega)
    end
end

"""
    transform_to_original_scale(y, spec::ParameterSpec)

Transform sampled values from the transformed scale back to the original
parameter scale.
"""
function transform_to_original_scale(y::AbstractVector, spec::ParameterSpec)
    if spec.transform == :logitnormal
        return bounded_logistic.(y, spec.lower, spec.upper)
    elseif spec.transform == :lognormal
        return exp.(y)
    else  # :normal
        return y
    end
end

"""
    generate_virtual_population(
        n::Int,
        param_specs::Vector{ParameterSpec},
        correlation_matrix::Matrix{Float64};
        seed::Union{Int,Nothing} = nothing
    ) -> DataFrame

Generate a virtual population of n plausible patients using Gaussian copulas.

# Algorithm (Figure 2 in paper):
1. Create marginal distributions on transformed scale
2. Generate correlated uniform samples using Gaussian copula
3. Transform uniforms to target marginals using inverse CDF
4. Transform from logit/log scale to original parameter scale

# Arguments
- `n`: Number of virtual patients to generate
- `param_specs`: Vector of ParameterSpec defining each parameter
- `correlation_matrix`: Correlation matrix (Spearman correlations preserved)
- `seed`: Random seed for reproducibility

# Returns
DataFrame with columns for each parameter and `id` column

# Example
```julia
params = TumorBurdenParams()
Rho = TumorBurdenCorrelation()
vpop = generate_virtual_population(10000, params, Rho; seed=22)
```
"""
function generate_virtual_population(
    n::Int,
    param_specs::Vector{ParameterSpec},
    correlation_matrix::Matrix{Float64};
    seed::Union{Int,Nothing} = nothing
)
    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_params = length(param_specs)
    @assert size(correlation_matrix) == (n_params, n_params) "Correlation matrix size mismatch"

    # Step 1: Create marginal distributions on transformed scale
    marginals = [create_marginal_distribution(spec) for spec in param_specs]

    # Step 2: Create Gaussian copula with correlation structure
    copula = GaussianCopula(correlation_matrix)

    # Step 3: Generate correlated uniform samples
    # Copulas.jl returns (n_params × n) matrix, transpose to (n × n_params)
    U = rand(copula, n)'

    # Step 4: Transform uniforms to target marginals (on transformed scale)
    Y = similar(U)
    for (j, marginal) in enumerate(marginals)
        Y[:, j] = quantile.(marginal, U[:, j])
    end

    # Step 5: Transform to original parameter scale
    P = similar(Y)
    for (j, spec) in enumerate(param_specs)
        P[:, j] = transform_to_original_scale(Y[:, j], spec)
    end

    # Create DataFrame with named columns
    df = DataFrame(P, [spec.name for spec in param_specs])
    df.id = 1:n

    # Reorder columns to put id first
    select!(df, :id, Not(:id))

    return df
end

"""
    generate_tumor_burden_vpop(n::Int; seed::Int = 22) -> DataFrame

Convenience function to generate virtual population for the Tumor Burden model.

# Example
```julia
vpop = generate_tumor_burden_vpop(10000)
```
"""
function generate_tumor_burden_vpop(n::Int; seed::Int = 22)
    params = TumorBurdenParams()
    Rho = TumorBurdenCorrelation()
    return generate_virtual_population(n, params, Rho; seed=seed)
end

#=============================================================================
# Validation Functions
=============================================================================#

"""
    validate_correlations(df::DataFrame, param_names::Vector{Symbol}, expected_corr::Matrix{Float64})

Validate that the sampled parameters have the expected Spearman correlations.

Uses StatsBase.corspearman for efficient computation.

Returns a NamedTuple with :observed and :expected correlation matrices.
"""
function validate_correlations(
    df::DataFrame,
    param_names::Vector{Symbol},
    expected_corr::Matrix{Float64};
    method::Symbol = :spearman
)
    param_matrix = Matrix(df[:, param_names])

    observed_corr = if method == :spearman
        corspearman(param_matrix)
    else
        cor(param_matrix)
    end

    max_diff = maximum(abs.(observed_corr - expected_corr))

    return (
        observed = observed_corr,
        expected = expected_corr,
        max_difference = max_diff,
        valid = max_diff < 0.05  # Allow 5% tolerance
    )
end

"""
    summarize_vpop(df::DataFrame, param_names::Vector{Symbol})

Generate summary statistics for the virtual population.
"""
function summarize_vpop(df::DataFrame, param_names::Vector{Symbol})
    summaries = DataFrame()

    for name in param_names
        values = df[!, name]
        push!(summaries, (
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
            q95 = quantile(values, 0.95)
        ))
    end

    return summaries
end

export ParameterSpec, TumorBurdenParams, TumorBurdenCorrelation
export generate_virtual_population, generate_tumor_burden_vpop
export validate_correlations, summarize_vpop
export bounded_logistic, logit_transform_mean
# Re-export from LogExpFunctions for convenience
export logit, logistic
