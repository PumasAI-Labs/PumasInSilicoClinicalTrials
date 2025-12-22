"""
    ISCTWorkflow

Julia/Pumas implementation of the In Silico Clinical Trial (ISCT) workflow
as described in the accompanying publication (DOI: 10.1002/psp4.70122).

This module is organized following the 7-step ISCT workflow from the paper:

    01-models/       Step 1-2: Mathematical Models & Parameterization
    02-sensitivity/  Step 3:   Sensitivity & Identifiability Analysis
    03-sampling/     Step 4:   Generate Plausible Patients (Vpop Generation)
    04-calibration/  Step 5:   VP Selection & Calibration (MILP)
    05-simulation/   Step 6:   Run In Silico Clinical Trial
    06-visualization/ Step 7:  Answer Questions of Interest (Analysis)

Two disease models are implemented:
- Tumor Burden Model (Section 3.1) - Simple 3-parameter example
- HBV QSP Model (Section 3.2) - Complex 11-ODE mechanistic model

Reference:
    Cortés-Ríos et al. "A Step-by-Step Workflow for Performing In Silico
    Clinical Trials With Nonlinear Mixed Effects Models"
    CPT: Pharmacometrics & Systems Pharmacology (2025)
"""
module ISCTWorkflow

using Reexport

# Core dependencies
using Pumas
using DataFrames
using DataFramesMeta
using Random

# Include submodules in workflow order (matches paper steps)

# Step 1-2: Mathematical Models & Parameterization
include("01-models/tumor_burden_model.jl")
include("01-models/hbv_model.jl")

# Step 3: Sensitivity & Identifiability Analysis
include("02-sensitivity/gsa_analysis.jl")

# Step 4: Parameter Sampling (Vpop Generation)
include("03-sampling/copula_sampling.jl")
include("03-sampling/hbv_sampling.jl")

# Step 5: VP Selection & Calibration
include("04-calibration/milp_calibration.jl")

# Step 6: Virtual Clinical Trial Simulation
include("05-simulation/vct_simulation.jl")

# Step 7: Analysis & Visualization
include("06-visualization/plotting.jl")

# Re-export Tumor Burden model components
# Note: _fixed variants removed - use zero_randeffs() or pass individual randeffs to simobs()
export tumor_burden_model, tumor_burden_model_analytical

# Re-export HBV model components
# Note: hbv_model_fixed removed - use zero_randeffs() pattern instead
export hbv_model, HBV_FIXED_PARAMS
export LOQ_HBsAg, LOQ_V, LOG_LOQ_V
export is_functional_cure, is_hbsag_loss

# Re-export Tumor Burden sampling components
export ParameterSpec, TumorBurdenParams, TumorBurdenCorrelation
export generate_virtual_population, generate_tumor_burden_vpop
export validate_correlations, summarize_vpop
export bounded_logistic, logit_transform_mean
export logit, logistic  # Re-exported from LogExpFunctions

# Re-export HBV sampling components
export HBV_ESTIMATED_PARAMS, HBV_FIXED_PARAM_NAMES, HBV_FIXED_VALUES
export load_hbv_vpop, extract_estimated_params, add_fixed_params
export HBVParameterStats, compute_hbv_stats, print_hbv_stats
export HBVParameterSpec, create_hbv_param_specs
export generate_hbv_vpop, generate_hbv_vpop_from_csv, subsample_hbv_vpop
export summarize_hbv_vpop

# Re-export MILP calibration components
export TargetDistribution, CalibrationResult, ParetoPoint
export create_target_from_data, create_target_from_specification
export classify_vps_to_bins, create_bin_membership_matrix
export solve_milp_calibration, solve_multivariable_calibration
export find_pareto_front, select_optimal_pareto_point
export calibrate_vpop, print_calibration_summary

# Re-export VCT simulation components
export TreatmentArm, CONTROL, NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO
export VCTConfig, VCTPatientResult, VCTResults
export get_phase_times, get_observation_times
export TUMOR_BURDEN_POP_PARAMS, compute_tumor_burden_randeffs
export simulate_tumor_burden_patient, run_tumor_burden_vct, run_tumor_burden_trial
export create_hbv_time_varying_covariates, simulate_hbv_patient, run_hbv_vct
export run_hbv_trial_comparison
export calculate_endpoint_rates, compare_treatment_arms, summarize_baseline_distribution

# HBV population dynamics exports
export simulate_hbv_dynamics, simulate_hbv_natural_history
export classify_hbv_outcome, summarize_hbv_dynamics_by_time

# Re-export GSA components
# Note: tumor_burden_gsa_model and hbv_gsa_model removed - use main models with constantcoef
export GSAParameterRange, GSAResult
export TUMOR_BURDEN_GSA_PARAMS, HBV_GSA_PARAMS
export run_tumor_burden_gsa, run_hbv_gsa
export create_param_ranges, create_constant_coef
export summarize_gsa, get_influential_params, print_gsa_summary

# Re-export Visualization components
export ISCT_THEME, TREATMENT_COLORS, set_isct_theme!
export HBV_OUTCOME_COLORS, HBV_BIOMARKER_LABELS, HBV_LOQ_THRESHOLDS
export plot_parameter_distributions, plot_parameter_correlations, plot_vpop_comparison
export plot_tumor_dynamics, plot_response_waterfall, plot_treatment_comparison
export plot_hbv_dynamics
# HBV population dynamics plots
export plot_hbv_population_dynamics, plot_hbv_natural_history, plot_hbv_treatment_response
export plot_hbv_biomarker_panel, add_treatment_phase_markers!
export plot_calibration_result, plot_pareto_front
export plot_gsa_indices, plot_gsa_heatmap, plot_gsa_comparison
export create_isct_summary_figure, save_figure, quick_hist, quick_scatter

#=============================================================================
# Virtual Clinical Trial Utilities
=============================================================================#

"""
    create_subjects_from_vpop(
        vpop::DataFrame,
        observation_times;
        treatment::Int = 1,
        id_col::Symbol = :id
    ) -> Vector{Subject}

Create Pumas Subject objects from a virtual population DataFrame.

# Arguments
- `vpop`: DataFrame with virtual patient parameters (columns: id, f, g, k, ...)
- `observation_times`: Time points for observations
- `treatment`: Treatment arm (0 = control, 1 = treatment)
- `id_col`: Column name for patient IDs

# Returns
Vector of Pumas Subject objects ready for simulation
"""
function create_subjects_from_vpop(
    vpop::DataFrame,
    observation_times;
    treatment::Int = 1,
    id_col::Symbol = :id
)
    subjects = Subject[]

    for row in eachrow(vpop)
        # Create subject with covariates
        subj = Subject(
            id = row[id_col],
            covariates = (treatment = treatment,),
            observations = (tumor_size = nothing,),
            time = observation_times
        )
        push!(subjects, subj)
    end

    return subjects
end

"""
    simulate_vpop(
        vpop::DataFrame,
        observation_times;
        treatment::Int = 1,
        seed::Union{Int,Nothing} = nothing,
        pop_params::NamedTuple = TUMOR_BURDEN_POP_PARAMS
    ) -> DataFrame

Simulate the virtual clinical trial for all patients in vpop.

Uses the main `tumor_burden_model` with individual parameters converted to random effects.
This follows Pumas best practices of using simobs(model, subj, params, randeffs).

# Arguments
- `vpop`: Virtual population DataFrame with parameters (f, g, k)
- `observation_times`: Vector of observation time points
- `treatment`: Treatment arm (0 = control, 1 = treatment)
- `seed`: Random seed for reproducibility
- `pop_params`: Population parameters (default: TUMOR_BURDEN_POP_PARAMS)

# Returns
DataFrame with simulation results: id, time, Nt (tumor size)
"""
function simulate_vpop(
    vpop::DataFrame,
    observation_times;
    treatment::Int = 1,
    seed::Union{Int,Nothing} = nothing,
    pop_params::NamedTuple = TUMOR_BURDEN_POP_PARAMS
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    results = DataFrame()

    for row in eachrow(vpop)
        # Create subject for this virtual patient
        subj = Subject(
            id = row.id,
            covariates = (treatment = treatment,),
            observations = (tumor_size = nothing,),
            time = observation_times
        )

        # Compute random effects from individual parameters
        randeffs = compute_tumor_burden_randeffs(
            row.f, row.g, row.k;
            tvf = pop_params.tvf,
            tvg = pop_params.tvg,
            tvk = pop_params.tvk
        )

        # Simulate with population parameters and individual random effects
        sim = simobs(tumor_burden_model, subj, pop_params, randeffs)

        # Extract results
        for (i, t) in enumerate(observation_times)
            push!(results, (
                id = row.id,
                time = t,
                Nt = sim.observations.Nt[i]
            ))
        end
    end

    return results
end

"""
    run_tumor_burden_vct(
        n_patients::Int;
        observation_times = 0:1:126,
        seed::Int = 22
    ) -> NamedTuple

Run a complete virtual clinical trial for the tumor burden model.

Uses the main `tumor_burden_model` with individual parameters converted to random effects,
following Pumas best practices.

Returns both treatment and control arm simulations along with the virtual population.

# Example
```julia
results = run_tumor_burden_vct(10000; seed=22)
results.vpop          # Virtual population parameters
results.treatment     # Treatment arm dynamics
results.control       # Control arm dynamics
```
"""
function run_tumor_burden_vct(
    n_patients::Int;
    observation_times = 0:1:126,
    seed::Int = 22
)
    # Generate virtual population
    vpop = generate_tumor_burden_vpop(n_patients; seed=seed)

    # Simulate treatment arm using main model with computed random effects
    treatment_results = simulate_vpop(
        vpop,
        collect(observation_times);
        treatment = 1
    )

    # Simulate control arm
    control_results = simulate_vpop(
        vpop,
        collect(observation_times);
        treatment = 0
    )

    return (
        vpop = vpop,
        treatment = treatment_results,
        control = control_results,
        observation_times = collect(observation_times)
    )
end

#=============================================================================
# Response Classification
=============================================================================#

"""
    classify_response(tumor_sizes::AbstractVector, threshold::Float64 = 0.7/5)

Classify patient response based on final tumor size.

According to the paper, complete response is defined as tumor size below
the minimum detectable threshold (typically ~7mm with baseline 3-5cm for Stage IIIA).

# Arguments
- `tumor_sizes`: Vector of tumor sizes at final time point
- `threshold`: Threshold for complete response (default: 0.14 = 7mm / 50mm baseline)

# Returns
Vector of response classifications (:complete_response or :no_response)
"""
function classify_response(tumor_sizes::AbstractVector; threshold::Float64 = 0.14)
    return [ts < threshold ? :complete_response : :no_response for ts in tumor_sizes]
end

"""
    calculate_response_rate(
        sim_results::DataFrame,
        time_weeks::Int;
        threshold::Float64 = 0.14
    ) -> Float64

Calculate the complete response rate at a specific time point.

# Arguments
- `sim_results`: Simulation results DataFrame with columns :id, :time, :Nt
- `time_weeks`: Time in weeks to evaluate response
- `threshold`: Threshold for complete response

# Returns
Response rate as a percentage (0-100)
"""
function calculate_response_rate(
    sim_results::DataFrame,
    time_weeks::Int;
    threshold::Float64 = 0.14
)
    time_days = time_weeks * 7

    # Get tumor sizes at specified time
    final_sizes = @chain sim_results begin
        @subset(:time .== time_days)
        @select(:id, :Nt)
    end

    n_responders = sum(final_sizes.Nt .< threshold)
    n_total = nrow(final_sizes)

    return 100.0 * n_responders / n_total
end

"""
    analyze_response_rates(
        sim_results::DataFrame;
        time_weeks::Vector{Int} = [6, 12, 18],
        threshold::Float64 = 0.14,
        n_bootstrap::Int = 100,
        bootstrap_size::Int = 1000
    ) -> DataFrame

Analyze response rates at multiple time points with bootstrap confidence intervals.

Replicates the analysis from Run_ISCT_results.m in the supplementary material.

# Returns
DataFrame with columns: week, median_rate, ci_lower (5th percentile), ci_upper (95th percentile)
"""
function analyze_response_rates(
    sim_results::DataFrame;
    time_weeks::Vector{Int} = [6, 12, 18],
    threshold::Float64 = 0.14,
    n_bootstrap::Int = 100,
    bootstrap_size::Int = 1000
)
    results = DataFrame(
        week = Int[],
        median_rate = Float64[],
        ci_lower = Float64[],
        ci_upper = Float64[]
    )

    unique_ids = unique(sim_results.id)
    n_patients = length(unique_ids)

    for week in time_weeks
        time_days = week * 7

        # Get tumor sizes at this time
        sizes_at_time = @chain sim_results begin
            @subset(:time .== time_days)
            @select(:id, :Nt)
        end

        # Bootstrap sampling
        rates = Float64[]
        for i in 1:n_bootstrap
            Random.seed!(i)
            sample_ids = rand(unique_ids, bootstrap_size)
            sample_sizes = [sizes_at_time[sizes_at_time.id .== id, :Nt][1] for id in sample_ids]
            rate = 100.0 * sum(sample_sizes .< threshold) / bootstrap_size
            push!(rates, rate)
        end

        push!(results, (
            week = week,
            median_rate = median(rates),
            ci_lower = quantile(rates, 0.05),
            ci_upper = quantile(rates, 0.95)
        ))
    end

    return results
end

export create_subjects_from_vpop, simulate_vpop, run_tumor_burden_vct
export classify_response, calculate_response_rate, analyze_response_rates

#=============================================================================
# HBV Virtual Clinical Trial Utilities
=============================================================================#

"""
Treatment phase timing constants (in days).
"""
const HBV_PHASES = (
    untreated = 5 * 365,      # 5 years untreated
    nuc_background = 4 * 365, # 4 years NA background (suppressed only)
    treatment = 48 * 7,       # 48 weeks treatment
    off_treatment = 24 * 7    # 24 weeks follow-up
)

"""
Treatment arm codes for HBV VCT.
"""
const HBV_TREATMENT = (
    control = 0,    # No treatment
    nuc = 1,        # NUC only
    ifn = 2,        # IFN only
    combo = 3       # NUC + IFN combination
)

"""
    create_hbv_dosing_schedule(
        treatment::Int;
        suppressed::Bool = false
    ) -> NamedTuple

Create dosing schedule for HBV virtual clinical trial.

Returns time vectors and dosing indicators for each phase.

# Arguments
- `treatment`: Treatment arm (0=control, 1=NUC, 2=IFN, 3=combo)
- `suppressed`: Whether patient is NA-suppressed at baseline

# Returns
NamedTuple with :times, :dNUC, :dIFN arrays
"""
function create_hbv_dosing_schedule(
    treatment::Int;
    suppressed::Bool = false
)
    # Define time spans for each phase
    t_untreated = 0:1:(HBV_PHASES.untreated)

    if suppressed
        t_nuc = (HBV_PHASES.untreated):(HBV_PHASES.untreated + HBV_PHASES.nuc_background)
    else
        t_nuc = (HBV_PHASES.untreated):(HBV_PHASES.untreated + HBV_PHASES.nuc_background)
    end

    t_start_tx = suppressed ? HBV_PHASES.untreated + HBV_PHASES.nuc_background : HBV_PHASES.untreated
    t_tx = t_start_tx:(t_start_tx + HBV_PHASES.treatment)
    t_off = (t_start_tx + HBV_PHASES.treatment):(t_start_tx + HBV_PHASES.treatment + HBV_PHASES.off_treatment)

    # Combine all time points
    all_times = sort(unique(vcat(collect(t_untreated), collect(t_nuc), collect(t_tx), collect(t_off))))

    # Create dosing vectors
    n_times = length(all_times)
    dNUC = zeros(n_times)
    dIFN = zeros(n_times)

    for (i, t) in enumerate(all_times)
        # NUC background (if suppressed)
        if suppressed && t > HBV_PHASES.untreated && t <= HBV_PHASES.untreated + HBV_PHASES.nuc_background
            dNUC[i] = 1.0
        end

        # Treatment phase
        if t > t_start_tx && t <= t_start_tx + HBV_PHASES.treatment
            if treatment == 1  # NUC only
                dNUC[i] = 1.0
            elseif treatment == 2  # IFN only
                dIFN[i] = 1.0
            elseif treatment == 3  # Combo
                dNUC[i] = 1.0
                dIFN[i] = 1.0
            end
        end
    end

    return (times = all_times, dNUC = dNUC, dIFN = dIFN)
end

"""
    calculate_hbv_fc_rate(
        sim_results::DataFrame;
        time_point::Symbol = :off_end,
        n_bootstrap::Int = 100,
        bootstrap_size::Int = 1000
    ) -> NamedTuple

Calculate functional cure rate from HBV simulation results.

# Arguments
- `sim_results`: DataFrame with :id, :log_HBsAg, :log_V at end of off-treatment
- `time_point`: Which time point to evaluate (:tx_end, :off_end)
- `n_bootstrap`: Number of bootstrap samples
- `bootstrap_size`: Size of each bootstrap sample

# Returns
NamedTuple with :rate, :ci_lower, :ci_upper
"""
function calculate_hbv_fc_rate(
    sim_results::DataFrame;
    n_bootstrap::Int = 100,
    bootstrap_size::Int = 1000
)
    n_total = nrow(sim_results)

    # Calculate FC status for each patient
    fc_status = [is_functional_cure(row.log_HBsAg, row.log_V) for row in eachrow(sim_results)]

    # Bootstrap sampling
    rates = Float64[]
    for i in 1:n_bootstrap
        Random.seed!(i)
        sample_indices = rand(1:n_total, bootstrap_size)
        rate = 100.0 * sum(fc_status[sample_indices]) / bootstrap_size
        push!(rates, rate)
    end

    return (
        rate = median(rates),
        ci_lower = quantile(rates, 0.025),
        ci_upper = quantile(rates, 0.975)
    )
end

"""
    calculate_hbv_hbsag_loss_rate(
        sim_results::DataFrame;
        n_bootstrap::Int = 100,
        bootstrap_size::Int = 1000
    ) -> NamedTuple

Calculate HBsAg loss rate from HBV simulation results.
"""
function calculate_hbv_hbsag_loss_rate(
    sim_results::DataFrame;
    n_bootstrap::Int = 100,
    bootstrap_size::Int = 1000
)
    n_total = nrow(sim_results)

    # Calculate HBsAg loss status
    loss_status = [is_hbsag_loss(row.log_HBsAg) for row in eachrow(sim_results)]

    # Bootstrap sampling
    rates = Float64[]
    for i in 1:n_bootstrap
        Random.seed!(i)
        sample_indices = rand(1:n_total, bootstrap_size)
        rate = 100.0 * sum(loss_status[sample_indices]) / bootstrap_size
        push!(rates, rate)
    end

    return (
        rate = median(rates),
        ci_lower = quantile(rates, 0.025),
        ci_upper = quantile(rates, 0.975)
    )
end

export HBV_PHASES, HBV_TREATMENT
export create_hbv_dosing_schedule
export calculate_hbv_fc_rate, calculate_hbv_hbsag_loss_rate

end # module
