# HBV Vpop Generation
# Helper script for generating HBV virtual populations
# Used by HBV-03 onwards

using Copulas
using DataFrames
using Pumas.Distributions
using Random
using LinearAlgebra

# HBV parameter specifications (log10 scale)
# 9 estimated parameters with IIV
const HBV_PARAM_SPECS = [
    (name = :beta,     tv = -5.0,  omega = 0.5),   # Infection rate
    (name = :p_S,      tv = 8.0,   omega = 0.4),   # HBsAg production
    (name = :m,        tv = 2.5,   omega = 0.3),   # Immune killing
    (name = :k_Z,      tv = -5.0,  omega = 0.4),   # ALT production
    (name = :convE,    tv = 2.5,   omega = 0.3),   # Effector conversion
    (name = :epsNUC,   tv = -2.0,  omega = 0.5),   # NUC efficacy
    (name = :epsIFN,   tv = -1.5,  omega = 0.6),   # IFN efficacy
    (name = :r_E_IFN,  tv = 0.3,   omega = 0.4),   # IFN immune boost
    (name = :k_D,      tv = -4.0,  omega = 0.4)    # DC activation
]

# Correlation matrix (identity for simplicity, can be updated with clinical data)
const HBV_CORRELATION = Matrix{Float64}(I, 9, 9)

"""
    generate_hbv_vpop(n; seed=nothing, correlation=HBV_CORRELATION)

Generate n virtual patients for the HBV model using Gaussian copula sampling.

# Arguments
- `n::Int`: Number of virtual patients to generate
- `seed::Union{Int,Nothing}`: Optional random seed for reproducibility
- `correlation::Matrix{Float64}`: Correlation matrix for copula

# Returns
- `DataFrame` with columns: id, beta, p_S, m, k_Z, convE, epsNUC, epsIFN, r_E_IFN, k_D
"""
function generate_hbv_vpop(n::Int;
                          seed::Union{Int,Nothing}=nothing,
                          correlation::Matrix{Float64}=HBV_CORRELATION)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_params = length(HBV_PARAM_SPECS)

    # Create Gaussian copula
    copula = GaussianCopula(correlation)
    U = rand(copula, n)'

    # Transform to target distributions
    params = similar(U)
    for j in 1:n_params
        tv = HBV_PARAM_SPECS[j].tv
        ω = HBV_PARAM_SPECS[j].omega
        # Sample from normal centered at tv with sd = omega
        eta = quantile.(Normal(0, ω), U[:, j])
        params[:, j] = tv .+ eta
    end

    return DataFrame(
        id = 1:n,
        beta = params[:, 1],
        p_S = params[:, 2],
        m = params[:, 3],
        k_Z = params[:, 4],
        convE = params[:, 5],
        epsNUC = params[:, 6],
        epsIFN = params[:, 7],
        r_E_IFN = params[:, 8],
        k_D = params[:, 9]
    )
end

"""
    calculate_baseline_hbsag(p_S)

Calculate baseline HBsAg (log10 IU/mL) from p_S parameter.

Simplified approximation based on steady-state relationship.
"""
function calculate_baseline_hbsag(p_S)
    # Baseline HBsAg approximation: log10 IU/mL
    # Based on steady-state relationship in HBV model
    return 3.0 + 0.5 * (p_S - 8.0)
end

"""
    add_baseline_hbsag!(vpop)

Add baseline HBsAg columns to vpop DataFrame.

Modifies vpop in place, adding:
- log_hbsag_bl: log10 HBsAg (IU/mL)
- hbsag_bl: HBsAg (IU/mL)
"""
function add_baseline_hbsag!(vpop::DataFrame)
    vpop[!, :log_hbsag_bl] = calculate_baseline_hbsag.(vpop.p_S)
    vpop[!, :hbsag_bl] = 10 .^ vpop.log_hbsag_bl
    return vpop
end

"""
    compute_hbv_stats(vpop)

Compute summary statistics for HBV virtual population.
"""
function compute_hbv_stats(vpop::DataFrame)
    param_names = [spec.name for spec in HBV_PARAM_SPECS]

    stats = DataFrame(
        parameter = param_names,
        mean = [mean(vpop[!, name]) for name in param_names],
        std = [std(vpop[!, name]) for name in param_names],
        median = [median(vpop[!, name]) for name in param_names],
        min = [minimum(vpop[!, name]) for name in param_names],
        max = [maximum(vpop[!, name]) for name in param_names]
    )

    return stats
end

# Export functions
export generate_hbv_vpop, HBV_PARAM_SPECS, HBV_CORRELATION
export calculate_baseline_hbsag, add_baseline_hbsag!, compute_hbv_stats
