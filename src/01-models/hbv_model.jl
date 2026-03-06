"""
    HBV QSP Model

Implements the 11-ODE HBV mechanistic model as described in Section 3.2
of the ISCT workflow paper.

Model captures:
- Hepatocyte populations (T: target, R: resistant, I: infected)
- Viral dynamics (V: virus, S: HBsAg surface antigen)
- Immune response (E: effector, Q, X, D, Y, Z)
- Treatment effects of NA (nucleoside analogs) and IFN (interferon)

Treatment arms:
- 0: Control (no treatment)
- 1: NUC only
- 2: IFN only
- 3: NUC + IFN combination

Simulation phases:
1. Untreated period (5 years)
2. NA background - if suppressed (4 years)
3. Treatment period (48 weeks)
4. Off-treatment period (24 weeks)

Parameters:
- 22 fixed parameters
- 9 estimated parameters with inter-individual variability
  (beta, p_S, m, k_Z, convE, epsNUC, epsIFN, r_E_IFN, k_D)
"""

using Pumas

#=============================================================================
# Fixed Parameter Values (from NLME fitting)
=============================================================================#

"""
Default fixed parameter values for the HBV model.
These are constants that don't vary between virtual patients.
"""
const HBV_FIXED_PARAMS = (
    iniV = -0.481486,      # Initial viral load (log10)
    p_V = 2.0,             # Viral production rate (log10)
    r_T = 1.0,             # Hepatocyte proliferation rate
    r_E = 0.1,             # Effector cell expansion rate
    T_max = 13600000.0,    # Maximum hepatocytes
    n = 2.0,               # Difference in killing rate
    phiE = 2.0,            # Half-saturation for immune activation (log10)
    dEtoX = 0.3,           # E to X conversion rate
    phiQ = 0.8,            # Q saturation constant
    d_V = 0.67,            # Virus clearance rate
    d_TI = 0.0039,         # Hepatocyte death rate
    d_E = -2.0,            # Effector cell death rate (log10)
    d_Z = -0.328,          # Z clearance rate (log10)
    d_Q = -2.414,          # Q clearance rate (log10)
    rho = -3.0,            # Cell resistance conversion (log10)
    r_X = 1.0,             # X growth rate
    d_X = 0.2,             # X death rate
    d_Y = 0.22,            # Y clearance rate
    Smax = 1.0,            # Maximum HBsAg suppression
    phiS = 0.147,          # HBsAg suppression saturation
    nS = 0.486,            # HBsAg suppression Hill coefficient
    d_D = -0.62157,        # D clearance rate (log10)
    iniZ = 1.25,            # Initial Z value (log10)
)

#=============================================================================
# HBV Model Definition
=============================================================================#

"""
    hbv_model

Pumas model for HBV viral dynamics with immune response and treatment effects.

State variables (11 ODEs):
- T: Target (uninfected) hepatocytes
- R: Resistant hepatocytes
- V: Virus (HBV DNA)
- S: HBsAg (surface antigen)
- Y: Dead cell marker
- Z: Immune response (ALT proxy)
- I: Infected hepatocytes
- D: Dendritic cell activation
- E: Effector T cells
- Q: Delayed effector signal
- X: Cytotoxic effect

Covariates:
- `treatment`: 0=control, 1=NUC, 2=IFN, 3=NUC+IFN
- `dNUC`: NUC dosing indicator (0 or 1)
- `dIFN`: IFN dosing indicator (0 or 1)
"""
hbv_model = @model begin
    @param begin
        # Fixed parameters (population values)
        iniV ∈ RealDomain(init = -0.481486)
        p_V ∈ RealDomain(init = 2.0)
        r_T ∈ RealDomain(lower = 0.0, init = 1.0)
        r_E ∈ RealDomain(lower = 0.0, init = 0.1)
        T_max ∈ RealDomain(lower = 0.0, init = 13600000.0)
        n ∈ RealDomain(lower = 0.0, init = 2.0)
        phiE ∈ RealDomain(init = 2.0)
        dEtoX ∈ RealDomain(lower = 0.0, init = 0.3)
        phiQ ∈ RealDomain(lower = 0.0, init = 0.8)
        d_V ∈ RealDomain(lower = 0.0, init = 0.67)
        d_TI ∈ RealDomain(lower = 0.0, init = 0.0039)
        d_E ∈ RealDomain(init = -2.0)
        d_Z ∈ RealDomain(init = -0.328)
        d_Q ∈ RealDomain(init = -2.414)
        rho ∈ RealDomain(init = -3.0)
        r_X ∈ RealDomain(lower = 0.0, init = 1.0)
        d_X ∈ RealDomain(lower = 0.0, init = 0.2)
        d_Y ∈ RealDomain(lower = 0.0, init = 0.22)
        Smax ∈ RealDomain(lower = 0.0, upper = 1.0, init = 0.5)
        phiS ∈ RealDomain(lower = 0.0, init = 0.147)
        nS ∈ RealDomain(lower = 0.0, init = 0.486)
        d_D ∈ RealDomain(init = -0.62157)
        iniZ ∈ RealDomain(init = 1.25)

        # Estimated parameters (with IIV)
        tvbeta ∈ RealDomain(init = -5.0)
        tvp_S ∈ RealDomain(init = 8.0)
        tvm ∈ RealDomain(init = 2.5)
        tvk_Z ∈ RealDomain(init = -5.0)
        tvconvE ∈ RealDomain(init = 2.5)
        tvepsNUC ∈ RealDomain(init = -2.0)
        tvepsIFN ∈ RealDomain(init = -1.5)
        tvr_E_IFN ∈ RealDomain(init = 0.3)
        tvk_D ∈ RealDomain(init = -4.0)

        # Inter-individual variability
        Ω ∈ PDiagDomain(9)

        # Residual error
        σ_HBsAg ∈ RealDomain(lower = 0.0, init = 0.1)
        σ_V ∈ RealDomain(lower = 0.0, init = 0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates dNUC dIFN

    @pre begin
        # Individual parameters - additive on log10-scale (achieves log-normal behavior)
        # Note: These parameters are used as 10^param in dynamics (e.g., 10^beta, 10^k_Z)
        beta = tvbeta + η[1]
        p_S = tvp_S + η[2]
        m = tvm + η[3]
        k_Z = tvk_Z + η[4]
        convE = tvconvE + η[5]
        epsNUC_ind = tvepsNUC + η[6]
        epsIFN_ind = tvepsIFN + η[7]
        r_E_IFN_ind = tvr_E_IFN + η[8]
        k_D = tvk_D + η[9]

        # Treatment effects
        # eps_NUC: effective NUC inhibition
        eps_NUC = dNUC > 0 ? (dNUC < 1 ? epsNUC_ind / dNUC : epsNUC_ind) : 0.0

        # eps_IFN and rr_E_IFN: effective IFN effects
        eps_IFN = dIFN > 0 ? (dIFN < 1 ? epsIFN_ind / dIFN : epsIFN_ind) : 0.0
        rr_E_IFN = dIFN > 0 ? (dIFN < 1 ? 10^(r_E_IFN_ind * dIFN) : 10^r_E_IFN_ind) : 1.0

        # Initial conditions
        V_0 = 10^iniV
        I_0 = d_V * V_0 / (10^p_V)
        T_0 = T_max - I_0
        S_0 = 10^p_S * I_0 / d_V
        Z_0 = 10^iniZ

        # Lambda for Z dynamics
        lambda_Z = (10^d_Z) * Z_0
    end

    @init begin
        T = T_0
        R = 0.0
        V = V_0
        S = S_0
        Y = 0.0
        Z = Z_0
        I = I_0
        D = 0.0
        E = 0.0
        Q = 0.0
        X = 0.0
    end

    @vars begin
        # Branch based on infection level (I >= 0.0001)
        # Using continuous approximation to avoid discontinuity
        infection_active = I / (I + 0.0001)
        # HBsAg level for X dynamics (μg/mL)
        SAg_ug_per_ml = (V + S) * (96 * 24000 * 1.0e6) / (6.023e23)
    end

    @dynamics begin
        # Infected hepatocytes
        I' = (10^beta) * T * V + r_T * I * (1 - (T + I + R) / T_max) -
            (10^m) * rr_E_IFN * E * I - (10^rho) * X * I - d_TI * I

        # Target hepatocytes
        T' = -(10^beta) * T * V + r_T * T * (1 - (T + I * infection_active + R) / T_max) -
            d_TI * T + (10^rho) * R / 100 - (10^(m - n)) * rr_E_IFN * E * T

        # Resistant hepatocytes
        R' = (10^rho) * X * I * infection_active + r_T * R * (1 - (T + I * infection_active + R) / T_max) -
            (10^rho) * R / 100 - d_TI * R - (10^(m - n)) * rr_E_IFN * E * R

        # Virus dynamics
        V' = (10^p_V) * 10^(eps_NUC + eps_IFN) * I * infection_active - d_V * V

        # HBsAg dynamics
        S' = (10^p_S) * I * infection_active - d_V * S

        # Dead cell marker
        Y' = (10^m) * rr_E_IFN * E * I * infection_active +
            (10^(m - n)) * rr_E_IFN * E * (T + R) - d_Y * Y

        # Immune response (ALT proxy)
        Z' = lambda_Z + (10^k_Z) * Y - (10^d_Z) * Z

        # Dendritic cell activation
        D' = (10^k_D) * I / (10^phiE + I) * infection_active - (10^d_D) * D

        # Effector T cells
        E' = (10^d_E + r_E * E) * D - dEtoX * E * Q^4 / (phiQ^4 + Q^4) - 10^d_E * E

        # Delayed effector signal
        Q' = 10^d_Q * D - 10^d_Q * Q

        # Cytotoxic effect
        X' = r_X * (1 - X) * (I / (10^phiE + I)) * infection_active *
            (1 - Smax * SAg_ug_per_ml^nS / (phiS^nS + SAg_ug_per_ml^nS)) - d_X * X
    end

    @derived begin
        # HBsAg in IU/mL (log10 scale)
        # Conversion: (V + S) * (96 * 24000 * 1e9) / (6.023e23 * 0.98)
        HBsAg_IU = @. (V + S) * (96 * 24000 * 1.0e9) / (6.023e23 * 0.98)
        log_HBsAg = @. log10(max(HBsAg_IU, 0.04))  # LOQ = 0.05 IU/mL

        # Viral load (log10 copies/mL)
        log_V = @. log10(max(V, 1.0e-6))

        # ALT proxy
        log_ALT = @. log10(max(Z, 1.0e-6))

        # Effector T cells (log10 scale for plotting)
        log_E = @. log10(max(E, 1.0e-6))

        # Observations with residual error
        HBsAg_obs ~ @. Normal(log_HBsAg, σ_HBsAg)
        V_obs ~ @. Normal(log_V, σ_V)
    end

    # @observed begin
    #     # GSA endpoints using NCA
    #     nca_hbsag := @nca log_HBsAg
    #     nca_viral := @nca log_V
    #     final_hbsag = NCA.clast(nca_hbsag)
    #     final_viral = NCA.clast(nca_viral)
    #     hbsag_nadir = NCA.cmin(nca_hbsag)
    # end
end

# Note: hbv_model_fixed has been removed.
# Use zero_randeffs(hbv_model, population, params) for deterministic simulations.
# Or pass individual parameters as random effects directly to simobs().

#=============================================================================
# Clinical Endpoints
=============================================================================#

"""
    LOQ_HBsAg

Limit of quantification for HBsAg: 0.05 IU/mL
Functional cure threshold.
"""
const LOQ_HBsAg = 0.05

"""
    LOQ_V

Limit of quantification for viral load: 25 copies/mL (log10 = 1.4)
"""
const LOQ_V = 25.0
const LOG_LOQ_V = log10(LOQ_V)

"""
    is_functional_cure(log_hbsag, log_v)

Determine if a patient has achieved functional cure.

Functional cure is defined as:
- HBsAg < 0.05 IU/mL (log10 < log10(0.05) ≈ -1.3)
- HBV DNA < 25 copies/mL (log10 < 1.4)

Both conditions must be met for ≥24 weeks post-treatment.
"""
function is_functional_cure(log_hbsag::Real, log_v::Real)
    return log_hbsag < log10(LOQ_HBsAg) && log_v < LOG_LOQ_V
end

"""
    is_hbsag_loss(log_hbsag)

Determine if a patient has achieved HBsAg loss.

HBsAg loss is defined as HBsAg < 0.05 IU/mL.
"""
function is_hbsag_loss(log_hbsag::Real)
    return log_hbsag < log10(LOQ_HBsAg)
end

export hbv_model, HBV_FIXED_PARAMS
export LOQ_HBsAg, LOQ_V, LOG_LOQ_V
export is_functional_cure, is_hbsag_loss


# s_no_treatment =
#   Subject(
#     id="No Treatment",
#     events=nothing,
#     # observations=(log10V=nothing, log10ALT=nothing, log10HBsAg=nothing),
#     observations=(log10V=nothing, log10HBs=nothing),
#     covariates=(; dNUC=fill(0, 26), dIFN=vcat(0, fill(0, 25)), dBepi=vcat(0, fill(0, 25)), wt = fill(73, 26) ),
#     covariates_time=vcat(0, 118:7:286)
# )

s_no_treatment =
    Subject(
    id = "No Treatment",
    events = nothing,
    # observations=(log10V=nothing, log10ALT=nothing, log10HBsAg=nothing),
    observations = (log10V = nothing, log10HBs = nothing),
    covariates = (; dNUC = 0, dIFN = 0, dBepi = 0, wt = 73)
)

#----------------------------------------------------------------------
s_treatment = Subject(
    id = "Treatment",
    events = nothing,
    # observations=(log10V=nothing, log10ALT=nothing, log10HBsAg=nothing),
    observations = (log10V = nothing, log10HBs = nothing),
    covariates = (; dNUC = vcat(0, fill(1, 25)), dIFN = vcat(0, fill(0, 25)), dBepi = vcat(0, fill(0, 25)), wt = fill(73, 26)),
    covariates_time = vcat(0, 118:7:286)
)
#zfx = zero_randeffs(hbv_model, pop_no_treatment, init_params(hbv_model))
#sim_pop4 = simobs(pk_27, pop4_sub, param, zfx, obstimes = 0.1:1:500)
using Random
rng = Random.seed!(1234)
using PumasUtilities
sims_no_treatment = simobs(
    hbv_model,
    s_no_treatment,
    init_params(hbv_model),
    #PARAMS,
    zero_randeffs(hbv_model, s_no_treatment, init_params(hbv_model)),
    rng = rng,
    obstimes = 0:0.5:365,
    simulate_error = false
)
sim_plot(
    sims_no_treatment,
    observations = [:log_V],
    figure = (; fontsize = 18),
    axis = (;
        xlabel = "Time (days)",
        ylabel = "Simulated V (No treatment)",
        xticks = 0:50:365,
    ),
)


sims_treatment = simobs(
    hbv_model,
    s_treatment,
    init_params(hbv_model),
    #PARAMS,
    zero_randeffs(hbv_model, s_no_treatment, init_params(hbv_model)),
    rng = rng,
    obstimes = 0:0.5:365,
    simulate_error = false
)
sim_plot(
    sims_treatment,
    observations = [:log_V],
    figure = (; fontsize = 18),
    axis = (;
        xlabel = "Time (days)",
        ylabel = "Simulated V (No treatment)",
        xticks = 0:50:365,
    ),
)


##
sims_steadystate = simobs(
    hbv_model, s_no_treatment, init_params(hbv_model),
    zero_randeffs(hbv_model, s_no_treatment, init_params(hbv_model)),
    obstimes = 0:1:1825  # 5 years
)
sim_plot(
    sims_steadystate,
    observations = [:log_V],
    figure = (; fontsize = 18),
    axis = (;
        xlabel = "Time (days)",
        ylabel = "Simulated V (No treatment)",
        xticks = 0:180:1825,
    ),
)
using AlgebraOfGraphics
using CairoMakie
using DataFramesMeta


"""
    plot_dynamic_symbol(df::DataFrame, sym::Symbol; kwargs...)

Plot a single dynamic symbol vs time from a simulation DataFrame.

# Arguments
- `df`: DataFrame from simobs containing time and the symbol column
- `sym`: Symbol name to plot (e.g., :Ab1, :Ag)
- `group_col`: Column to group/color by (default: :id)
- `time_col`: Time column name (default: :time)
- `kwargs...`: Additional arguments passed to draw()

# Returns
- AlgebraOfGraphics Figure object
"""
function plot_dynamic_symbol(
        df::DataFrame,
        sym::Symbol;
        group_col::Symbol = :id,
        time_col::Symbol = :time,
        kwargs...
    )
    sym_str = String(sym)
    if sym_str ∉ names(df)
        error("Symbol $sym not found in DataFrame. Available columns: $(names(df))")
    end

    plt = data(df) *
        mapping(time_col, sym => sym_str; color = group_col => nonnumeric) *
        visual(Lines; alpha = 0.7)

    fig = draw(plt; axis = (xlabel = "Time", ylabel = sym_str, title = sym_str), kwargs...)
    return fig
end

"""
    plot_all_dynamic_symbols(df::DataFrame, model; kwargs...)

Plot all available dynamic symbols from a model vs time.

# Arguments
- `df`: DataFrame from simobs
- `model`: Pumas model object with .syms.dynamic field
- `save_dir`: Optional directory to save plots (default: nothing, no saving)
- `kwargs...`: Additional arguments passed to plot_dynamic_symbol

# Returns
- Dictionary of Symbol => Figure pairs
"""
function plot_all_dynamic_symbols(
        df::DataFrame,
        model;
        save_dir::Union{String, Nothing} = nothing,
        kwargs...
    )
    dynamic_syms = collect(model.syms.dynamic)
    available_syms = [s for s in dynamic_syms if String(s) in names(df)]

    if isempty(available_syms)
        @warn "No dynamic symbols found in DataFrame"
        return Dict{Symbol, Any}()
    end

    println("Plotting $(length(available_syms)) dynamic symbols: $(available_syms)")

    figures = Dict{Symbol, Any}()
    for sym in available_syms
        try
            fig = plot_dynamic_symbol(df, sym; kwargs...)
            figures[sym] = fig

            if !isnothing(save_dir)
                mkpath(save_dir)
                save(joinpath(save_dir, "$(sym)_vs_time.png"), fig)
            end
        catch e
            @warn "Failed to plot $sym: $e"
        end
    end

    return figures
end

"""
    plot_dynamic_symbols_grid(df::DataFrame, model; ncols = 4, kwargs...)

Plot all dynamic symbols in a single figure grid layout.

# Arguments
- `df`: DataFrame from simobs
- `model`: Pumas model object
- `ncols`: Number of columns in the grid (default: 4)
- `figsize`: Figure size as (width, height) tuple (default: (1600, 1200))
- `kwargs...`: Additional arguments

# Returns
- Single Figure with all dynamic symbol plots
"""
function plot_dynamic_symbols_grid(
        df::DataFrame,
        model;
        ncols::Int = 4,
        figsize::Tuple{Int, Int} = (1600, 1200),
        group_col::Symbol = :id,
        time_col::Symbol = :time
    )
    dynamic_syms = collect(model.syms.dynamic)
    available_syms = [s for s in dynamic_syms if String(s) in names(df)]

    n = length(available_syms)
    nrows = ceil(Int, n / ncols)

    fig = Figure(size = figsize)

    for (i, sym) in enumerate(available_syms)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1

        ax = Axis(
            fig[row, col],
            xlabel = "Time",
            ylabel = String(sym),
            title = String(sym)
        )

        # Group by id and plot each separately
        for gdf in groupby(df, group_col)
            lines!(ax, gdf[:, time_col], gdf[:, sym]; alpha = 0.7)
        end
    end

    return fig
end
# Plot all in a grid layout
simdf = DataFrame(sims_no_treatment)
grid_fig = plot_dynamic_symbols_grid(simdf, hbv_model; ncols = 3, figsize = (2000, 1600))
