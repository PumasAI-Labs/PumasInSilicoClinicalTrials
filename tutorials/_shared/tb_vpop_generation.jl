# Tumor Burden Vpop Generation
# Helper script for generating tumor burden virtual populations
# Used by TB-03 onwards

using Copulas
using DataFrames
using Pumas.Distributions
using LogExpFunctions
using Random

# Parameter specifications from Qi & Cao 2023
const TB_PARAM_SPECS = [
    (name = :f, median = 0.27, omega = 2.16, lower = 0.0, upper = 1.0),
    (name = :g, median = 0.0013, omega = 1.57, lower = 0.0, upper = 0.13),
    (name = :k, median = 0.0091, omega = 1.24, lower = 0.0, upper = 1.6),
]

# Correlation matrix: r(f,g) = -0.64
const TB_CORRELATION = [
    1.0   -0.64  0.0
    -0.64   1.0   0.0
    0.0    0.0   1.0
]

"""
    logit_mean(P, a, b)

Calculate the logit of the proportion (P-a)/(b-a).
"""
function logit_mean(P, a, b)
    return logit((P - a) / (b - a))
end

"""
    bounded_logistic(y, a, b)

Apply bounded logistic transformation: a + (b-a) * logistic(y)
"""
function bounded_logistic(y, a, b)
    return a + (b - a) * logistic(y)
end

"""
    generate_tb_vpop(n; seed=nothing)

Generate n virtual patients for the tumor burden model using Gaussian copula sampling.

# Arguments
- `n::Int`: Number of virtual patients to generate
- `seed::Union{Int,Nothing}`: Optional random seed for reproducibility

# Returns
- `DataFrame` with columns: id, f, g, k
"""
function generate_tb_vpop(n::Int; seed::Union{Int, Nothing} = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Create copula and sample
    copula = GaussianCopula(TB_CORRELATION)
    U = rand(copula, n)'

    # Transform to target distributions
    n_params = length(TB_PARAM_SPECS)
    μ_logit = [logit_mean(s.median, s.lower, s.upper) for s in TB_PARAM_SPECS]

    Y = similar(U)
    for j in 1:n_params
        marginal = Normal(μ_logit[j], TB_PARAM_SPECS[j].omega)
        Y[:, j] = quantile.(marginal, U[:, j])
    end

    # Transform to original scale
    P = similar(Y)
    for (j, spec) in enumerate(TB_PARAM_SPECS)
        P[:, j] = bounded_logistic.(Y[:, j], spec.lower, spec.upper)
    end

    return DataFrame(
        id = 1:n,
        f = P[:, 1],
        g = P[:, 2],
        k = P[:, 3]
    )
end

# Export functions
export generate_tb_vpop, TB_PARAM_SPECS, TB_CORRELATION
