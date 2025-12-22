"""
    Virtual Clinical Trial Simulation Framework

This module implements Step 6 of the ISCT workflow:
"Run In Silico Clinical Trial"

The framework supports:
1. Multi-phase simulation (untreated → NUC background → treatment → off-treatment)
2. Multiple treatment arms (control, NUC, IFN, NUC+IFN)
3. Batch simulation of virtual populations
4. Result collection at key time points
5. Clinical endpoint analysis

Reference:
    Section 2, Step 6 of the ISCT workflow paper
"""

using Pumas
using DataFrames
using DataFramesMeta
using Statistics
using Random
using LinearAlgebra: Diagonal
# using ProgressMeter

#=============================================================================
# Treatment Configuration
=============================================================================#

"""
Treatment arm enumeration for VCT.
"""
@enum TreatmentArm begin
    CONTROL = 0
    NUC_ONLY = 1
    IFN_ONLY = 2
    NUC_IFN_COMBO = 3
end

"""
    VCTConfig

Configuration for a virtual clinical trial.

# Fields
- `treatment::TreatmentArm`: Treatment arm
- `suppressed::Bool`: Whether patients are NA-suppressed at baseline
- `untreated_duration::Int`: Days of untreated period
- `nuc_background_duration::Int`: Days of NUC background (suppressed only)
- `treatment_duration::Int`: Days of treatment
- `off_treatment_duration::Int`: Days of off-treatment follow-up
- `observation_interval::Int`: Days between observations
"""
struct VCTConfig
    treatment::TreatmentArm
    suppressed::Bool
    untreated_duration::Int
    nuc_background_duration::Int
    treatment_duration::Int
    off_treatment_duration::Int
    observation_interval::Int

    function VCTConfig(;
        treatment::TreatmentArm = NUC_IFN_COMBO,
        suppressed::Bool = false,
        untreated_duration::Int = 5 * 365,
        nuc_background_duration::Int = 4 * 365,
        treatment_duration::Int = 48 * 7,
        off_treatment_duration::Int = 24 * 7,
        observation_interval::Int = 7
    )
        new(treatment, suppressed, untreated_duration, nuc_background_duration,
            treatment_duration, off_treatment_duration, observation_interval)
    end
end

"""
    get_phase_times(config::VCTConfig) -> NamedTuple

Calculate the start and end times for each simulation phase.
"""
function get_phase_times(config::VCTConfig)
    t_untreated_start = 0
    t_untreated_end = config.untreated_duration

    if config.suppressed
        t_nuc_start = t_untreated_end
        t_nuc_end = t_nuc_start + config.nuc_background_duration
        t_tx_start = t_nuc_end
    else
        t_nuc_start = t_untreated_end
        t_nuc_end = t_untreated_end
        t_tx_start = t_untreated_end
    end

    t_tx_end = t_tx_start + config.treatment_duration
    t_off_start = t_tx_end
    t_off_end = t_off_start + config.off_treatment_duration

    return (
        untreated = (start=t_untreated_start, stop=t_untreated_end),
        nuc_background = (start=t_nuc_start, stop=t_nuc_end),
        treatment = (start=t_tx_start, stop=t_tx_end),
        off_treatment = (start=t_off_start, stop=t_off_end),
        total_duration = t_off_end
    )
end

"""
    get_observation_times(config::VCTConfig) -> Vector{Int}

Get the observation time points for the VCT.
"""
function get_observation_times(config::VCTConfig)
    phases = get_phase_times(config)
    return collect(0:config.observation_interval:phases.total_duration)
end

#=============================================================================
# VCT Results
=============================================================================#

"""
    VCTPatientResult

Simulation results for a single virtual patient.
"""
struct VCTPatientResult
    id::Int
    # Key time points (log10 values)
    baseline_hbsag::Float64
    baseline_viral::Float64
    end_nuc_hbsag::Float64
    end_nuc_viral::Float64
    end_tx_hbsag::Float64
    end_tx_viral::Float64
    end_off_hbsag::Float64
    end_off_viral::Float64
    # Endpoints
    hbsag_loss_tx::Bool      # HBsAg < LOQ at end of treatment
    hbsag_loss_off::Bool     # HBsAg < LOQ at end of off-treatment
    functional_cure::Bool    # Both HBsAg and Viral < LOQ at end of off-treatment
    # Status
    integration_error::Bool
end

"""
    VCTResults

Results from a complete virtual clinical trial.
"""
struct VCTResults
    config::VCTConfig
    patient_results::Vector{VCTPatientResult}
    dynamics::DataFrame  # Full time series (optional)
    summary::DataFrame   # Summary statistics
end

#=============================================================================
# Tumor Burden VCT Simulation
=============================================================================#

# Default population parameters for tumor burden model
const TUMOR_BURDEN_POP_PARAMS = (
    tvf = 0.27,
    tvg = 0.0013,
    tvk = 0.0091,
    Ω = Diagonal([2.16, 1.57, 1.24]),
    σ = 0.05
)

"""
    compute_tumor_burden_randeffs(f, g, k; tvf=0.27, tvg=0.0013, tvk=0.0091) -> NamedTuple

Compute random effects (η) from individual parameter values.

The tumor burden model uses:
- f = tvf * exp(η[1]) / (1 + tvf * (exp(η[1]) - 1))  (logit-normal)
- g = tvg * exp(η[2])  (log-normal)
- k = tvk * exp(η[3])  (log-normal)

This function inverts these relationships to find η values.
"""
function compute_tumor_burden_randeffs(f, g, k; tvf=0.27, tvg=0.0013, tvk=0.0091)
    # For log-normal: param = tv * exp(η) → η = log(param/tv)
    η2 = log(g / tvg)
    η3 = log(k / tvk)

    # For logit-normal: f = tvf * x / (1 + tvf * (x - 1)) where x = exp(η[1])
    # Solving: x = f * (1 - tvf) / (tvf * (1 - f))
    # Handle edge cases for f near 0 or 1
    f_clamped = clamp(f, 1e-10, 1.0 - 1e-10)
    x = f_clamped * (1.0 - tvf) / (tvf * (1.0 - f_clamped))
    η1 = log(max(x, 1e-10))

    return (η = [η1, η2, η3],)
end

"""
    simulate_tumor_burden_patient(
        model,
        pop_params::NamedTuple,
        randeffs::NamedTuple,
        observation_times::Vector;
        treatment::Int = 1
    ) -> DataFrame

Simulate a single tumor burden patient using population parameters and individual random effects.
"""
function simulate_tumor_burden_patient(
    model,
    pop_params::NamedTuple,
    randeffs::NamedTuple,
    observation_times::Vector;
    treatment::Int = 1
)
    # Create subject
    subj = Subject(
        id = 1,
        covariates = (treatment = treatment,),
        observations = (tumor_size = nothing,),
        time = observation_times
    )

    # Simulate with population parameters and individual random effects
    sim = simobs(model, subj, pop_params, randeffs)

    # Extract results
    return DataFrame(
        time = observation_times,
        Nt = sim.observations.Nt
    )
end

"""
    run_tumor_burden_vct(
        vpop::DataFrame,
        observation_times::Vector;
        treatment::Int = 1,
        seed::Union{Int,Nothing} = nothing,
        show_progress::Bool = true,
        pop_params::NamedTuple = TUMOR_BURDEN_POP_PARAMS
    ) -> DataFrame

Run tumor burden VCT for an entire virtual population.

Uses the main `tumor_burden_model` with individual parameters converted to random effects.
This follows Pumas best practices of using a single model with simobs(model, subj, params, randeffs).

# Arguments
- `vpop`: Virtual population with columns :id, :f, :g, :k
- `observation_times`: Time points for observation
- `treatment`: Treatment arm (0=control, 1=treatment)
- `seed`: Random seed
- `show_progress`: Whether to show progress bar
- `pop_params`: Population parameters (default: TUMOR_BURDEN_POP_PARAMS)

# Returns
DataFrame with columns: id, time, Nt, treatment
"""
function run_tumor_burden_vct(
    vpop::DataFrame,
    observation_times::Vector;
    treatment::Int = 1,
    seed::Union{Int,Nothing} = nothing,
    show_progress::Bool = true,
    pop_params::NamedTuple = TUMOR_BURDEN_POP_PARAMS
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_patients = nrow(vpop)
    results = DataFrame()

    # Use the main model (not _fixed variant)
    # Individual parameters are converted to random effects
    model = tumor_burden_model

    # iter = show_progress ? ProgressBar(1:n_patients) : 1:n_patients
    iter = 1:n_patients
    for i in iter
        row = vpop[i, :]

        # Compute random effects from individual parameters
        randeffs = compute_tumor_burden_randeffs(
            row.f, row.g, row.k;
            tvf = pop_params.tvf,
            tvg = pop_params.tvg,
            tvk = pop_params.tvk
        )

        patient_results = simulate_tumor_burden_patient(
            model, pop_params, randeffs, observation_times; treatment = treatment
        )
        patient_results.id .= row.id
        patient_results.treatment .= treatment

        append!(results, patient_results)
    end

    return results
end

"""
    run_tumor_burden_trial(
        vpop::DataFrame;
        observation_times = collect(0:1:126),
        seed::Int = 22,
        show_progress::Bool = true
    ) -> NamedTuple

Run a complete tumor burden trial with treatment and control arms.

# Returns
NamedTuple with :treatment, :control, :vpop, :config
"""
function run_tumor_burden_trial(
    vpop::DataFrame;
    observation_times = collect(0:1:126),
    seed::Int = 22,
    show_progress::Bool = true
)
    println("Running Treatment Arm...")
    treatment_results = run_tumor_burden_vct(
        vpop, observation_times;
        treatment = 1, seed = seed, show_progress = show_progress
    )

    println("Running Control Arm...")
    control_results = run_tumor_burden_vct(
        vpop, observation_times;
        treatment = 0, seed = seed, show_progress = show_progress
    )

    return (
        treatment = treatment_results,
        control = control_results,
        vpop = vpop,
        observation_times = observation_times
    )
end

#=============================================================================
# HBV Multi-Phase Simulation
=============================================================================#

"""
    HBVSimulationState

State of an HBV simulation at a given time point.
"""
struct HBVSimulationState
    time::Float64
    T::Float64      # Target hepatocytes
    R::Float64      # Resistant hepatocytes
    V::Float64      # Virus
    S::Float64      # HBsAg
    Y::Float64      # Dead cell marker
    Z::Float64      # Immune response (ALT)
    I::Float64      # Infected hepatocytes
    D::Float64      # Dendritic cells
    E::Float64      # Effector T cells
    Q::Float64      # Delayed signal
    X::Float64      # Cytotoxic effect
end

"""
    create_hbv_time_varying_covariates(
        config::VCTConfig,
        observation_times::Vector{Int}
    ) -> DataFrame

Create time-varying covariate DataFrame for HBV simulation.
"""
function create_hbv_time_varying_covariates(
    config::VCTConfig,
    observation_times::Vector{Int}
)
    phases = get_phase_times(config)
    n_times = length(observation_times)

    dNUC = zeros(n_times)
    dIFN = zeros(n_times)

    for (i, t) in enumerate(observation_times)
        # NUC background phase (if suppressed)
        if config.suppressed && t > phases.untreated.stop && t <= phases.nuc_background.stop
            dNUC[i] = 1.0
        end

        # Treatment phase
        if t > phases.treatment.start && t <= phases.treatment.stop
            if config.treatment == NUC_ONLY || config.treatment == NUC_IFN_COMBO
                dNUC[i] = 1.0
            end
            if config.treatment == IFN_ONLY || config.treatment == NUC_IFN_COMBO
                dIFN[i] = 1.0
            end
        end
    end

    return DataFrame(
        time = observation_times,
        dNUC = dNUC,
        dIFN = dIFN
    )
end

"""
    simulate_hbv_patient(
        params::NamedTuple,
        config::VCTConfig;
        observation_times::Union{Nothing,Vector} = nothing
    ) -> VCTPatientResult

Simulate a single HBV patient through all trial phases.

Note: This is a simplified simulation that doesn't use the full ODE model.
For production use, integrate with Pumas simobs().
"""
function simulate_hbv_patient(
    id::Int,
    params::NamedTuple,
    config::VCTConfig;
    observation_times::Union{Nothing,Vector} = nothing
)
    phases = get_phase_times(config)

    if isnothing(observation_times)
        observation_times = get_observation_times(config)
    end

    # Extract key parameters
    iniV = get(params, :iniV, -0.481486)
    p_S = params.p_S
    p_V = get(params, :p_V, 2.0)
    d_V = get(params, :d_V, 0.67)
    T_max = get(params, :T_max, 13600000.0)

    # Calculate initial conditions
    V_0 = 10^iniV
    I_0 = d_V * V_0 / (10^p_V)
    S_0 = 10^p_S * I_0 / d_V

    # HBsAg in IU/mL (log10)
    hbsag_baseline = log10((V_0 + S_0) * (96 * 24000 * 1e9) / (6.023e23 * 0.98))
    viral_baseline = log10(max(V_0, 1e-6))

    # Simplified dynamics simulation
    # In a full implementation, this would use ODE integration
    # Here we use a simplified exponential decay model for demonstration

    # Treatment efficacy
    eps_nuc = params.epsNUC
    eps_ifn = params.epsIFN

    # Calculate approximate endpoints based on treatment
    treatment_factor = 1.0
    if config.treatment == NUC_ONLY
        treatment_factor = 10^eps_nuc
    elseif config.treatment == IFN_ONLY
        treatment_factor = 10^eps_ifn
    elseif config.treatment == NUC_IFN_COMBO
        treatment_factor = 10^(eps_nuc + eps_ifn)
    end

    # Approximate HBsAg at key time points
    # This is a simplified model - real simulation would use full ODEs
    decay_rate = -log(treatment_factor) / config.treatment_duration

    hbsag_end_nuc = hbsag_baseline  # No change during NUC background for HBsAg
    viral_end_nuc = config.suppressed ? 1.4 : viral_baseline  # Suppressed if on NUC

    # Treatment effect
    if config.treatment != CONTROL
        hbsag_end_tx = hbsag_baseline - decay_rate * config.treatment_duration / 100
        viral_end_tx = viral_end_nuc - decay_rate * config.treatment_duration / 50
    else
        hbsag_end_tx = hbsag_baseline
        viral_end_tx = viral_baseline
    end

    # Off-treatment (potential rebound)
    rebound_factor = config.treatment == CONTROL ? 0.0 : 0.1
    hbsag_end_off = hbsag_end_tx + rebound_factor
    viral_end_off = viral_end_tx + rebound_factor

    # Calculate endpoints
    LOQ_HBsAg = log10(0.05)
    LOQ_V = 1.4  # log10(25)

    hbsag_loss_tx = hbsag_end_tx < LOQ_HBsAg
    hbsag_loss_off = hbsag_end_off < LOQ_HBsAg
    functional_cure = hbsag_end_off < LOQ_HBsAg && viral_end_off < LOQ_V

    return VCTPatientResult(
        id,
        hbsag_baseline, viral_baseline,
        hbsag_end_nuc, viral_end_nuc,
        hbsag_end_tx, viral_end_tx,
        hbsag_end_off, viral_end_off,
        hbsag_loss_tx, hbsag_loss_off, functional_cure,
        false  # No integration error in simplified model
    )
end

"""
    run_hbv_vct(
        vpop::DataFrame,
        config::VCTConfig;
        seed::Union{Int,Nothing} = nothing,
        show_progress::Bool = true
    ) -> VCTResults

Run HBV virtual clinical trial for an entire population.
"""
function run_hbv_vct(
    vpop::DataFrame,
    config::VCTConfig;
    seed::Union{Int,Nothing} = nothing,
    show_progress::Bool = true
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_patients = nrow(vpop)
    patient_results = Vector{VCTPatientResult}(undef, n_patients)

    # iter = show_progress ? ProgressBar(1:n_patients) : 1:n_patients
    iter = 1:n_patients

    for i in iter
        row = vpop[i, :]

        # Build parameter tuple
        params = (
            beta = row.beta,
            p_S = row.p_S,
            m = row.m,
            k_Z = row.k_Z,
            convE = row.convE,
            epsNUC = row.epsNUC,
            epsIFN = row.epsIFN,
            r_E_IFN = row.r_E_IFN,
            k_D = row.k_D,
            # Add fixed parameters if present
            iniV = hasproperty(vpop, :iniV) ? row.iniV : -0.481486,
            p_V = hasproperty(vpop, :p_V) ? row.p_V : 2.0,
            d_V = hasproperty(vpop, :d_V) ? row.d_V : 0.67,
            T_max = hasproperty(vpop, :T_max) ? row.T_max : 13600000.0
        )

        patient_results[i] = simulate_hbv_patient(row.id, params, config)
    end

    # Create summary DataFrame
    summary = DataFrame(
        id = [r.id for r in patient_results],
        baseline_hbsag = [r.baseline_hbsag for r in patient_results],
        baseline_viral = [r.baseline_viral for r in patient_results],
        end_nuc_hbsag = [r.end_nuc_hbsag for r in patient_results],
        end_nuc_viral = [r.end_nuc_viral for r in patient_results],
        end_tx_hbsag = [r.end_tx_hbsag for r in patient_results],
        end_tx_viral = [r.end_tx_viral for r in patient_results],
        end_off_hbsag = [r.end_off_hbsag for r in patient_results],
        end_off_viral = [r.end_off_viral for r in patient_results],
        hbsag_loss_tx = [r.hbsag_loss_tx for r in patient_results],
        hbsag_loss_off = [r.hbsag_loss_off for r in patient_results],
        functional_cure = [r.functional_cure for r in patient_results],
        integration_error = [r.integration_error for r in patient_results]
    )

    return VCTResults(config, patient_results, DataFrame(), summary)
end

#=============================================================================
# Trial Comparison
=============================================================================#

"""
    run_hbv_trial_comparison(
        vpop::DataFrame;
        treatments::Vector{TreatmentArm} = [CONTROL, NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO],
        suppressed::Bool = false,
        seed::Int = 42
    ) -> Dict{TreatmentArm, VCTResults}

Run multiple treatment arms for comparison.
"""
function run_hbv_trial_comparison(
    vpop::DataFrame;
    treatments::Vector{TreatmentArm} = [CONTROL, NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO],
    suppressed::Bool = false,
    seed::Int = 42
)
    results = Dict{TreatmentArm, VCTResults}()

    for treatment in treatments
        println("Running $(treatment) arm...")
        config = VCTConfig(treatment=treatment, suppressed=suppressed)
        results[treatment] = run_hbv_vct(vpop, config; seed=seed)
    end

    return results
end

#=============================================================================
# Result Analysis
=============================================================================#

"""
    calculate_endpoint_rates(
        results::VCTResults;
        n_bootstrap::Int = 100,
        bootstrap_size::Int = 1000
    ) -> DataFrame

Calculate endpoint rates with bootstrap confidence intervals.
"""
function calculate_endpoint_rates(
    results::VCTResults;
    n_bootstrap::Int = 100,
    bootstrap_size::Int = 1000
)
    summary = results.summary
    n_total = nrow(summary)

    endpoints = [:hbsag_loss_tx, :hbsag_loss_off, :functional_cure]
    endpoint_labels = ["HBsAg Loss (End Tx)", "HBsAg Loss (Off Tx)", "Functional Cure"]

    rates_df = DataFrame(
        endpoint = String[],
        rate = Float64[],
        ci_lower = Float64[],
        ci_upper = Float64[],
        n_responders = Int[],
        n_total = Int[]
    )

    for (endpoint, label) in zip(endpoints, endpoint_labels)
        endpoint_values = summary[!, endpoint]

        # Bootstrap
        rates = Float64[]
        for i in 1:n_bootstrap
            Random.seed!(i)
            sample_indices = rand(1:n_total, bootstrap_size)
            rate = 100.0 * sum(endpoint_values[sample_indices]) / bootstrap_size
            push!(rates, rate)
        end

        push!(rates_df, (
            endpoint = label,
            rate = median(rates),
            ci_lower = quantile(rates, 0.025),
            ci_upper = quantile(rates, 0.975),
            n_responders = sum(endpoint_values),
            n_total = n_total
        ))
    end

    return rates_df
end

"""
    compare_treatment_arms(
        trial_results::Dict{TreatmentArm, VCTResults}
    ) -> DataFrame

Compare endpoint rates across treatment arms.
"""
function compare_treatment_arms(
    trial_results::Dict{TreatmentArm, VCTResults}
)
    comparison = DataFrame()

    for (treatment, results) in trial_results
        rates = calculate_endpoint_rates(results)
        rates.treatment .= string(treatment)
        append!(comparison, rates)
    end

    return comparison
end

"""
    summarize_baseline_distribution(results::VCTResults) -> DataFrame

Summarize the baseline distribution of the virtual population.
"""
function summarize_baseline_distribution(results::VCTResults)
    summary = results.summary

    stats = DataFrame(
        variable = ["HBsAg (log10 IU/mL)", "Viral Load (log10 copies/mL)"],
        mean = [mean(summary.baseline_hbsag), mean(summary.baseline_viral)],
        median = [median(summary.baseline_hbsag), median(summary.baseline_viral)],
        std = [std(summary.baseline_hbsag), std(summary.baseline_viral)],
        min = [minimum(summary.baseline_hbsag), minimum(summary.baseline_viral)],
        max = [maximum(summary.baseline_hbsag), maximum(summary.baseline_viral)],
        q05 = [quantile(summary.baseline_hbsag, 0.05), quantile(summary.baseline_viral, 0.05)],
        q95 = [quantile(summary.baseline_hbsag, 0.95), quantile(summary.baseline_viral, 0.95)]
    )

    return stats
end

#=============================================================================
# Exports
=============================================================================#

export TreatmentArm, CONTROL, NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO
export VCTConfig, VCTPatientResult, VCTResults
export get_phase_times, get_observation_times
export TUMOR_BURDEN_POP_PARAMS, compute_tumor_burden_randeffs
export simulate_tumor_burden_patient, run_tumor_burden_vct, run_tumor_burden_trial
export create_hbv_time_varying_covariates, simulate_hbv_patient, run_hbv_vct
export run_hbv_trial_comparison
export calculate_endpoint_rates, compare_treatment_arms, summarize_baseline_distribution
