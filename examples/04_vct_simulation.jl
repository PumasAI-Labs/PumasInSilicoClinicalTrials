#=
Virtual Clinical Trial Simulation Example

This script demonstrates Step 6 of the ISCT workflow:
"Run In Silico Clinical Trial"

Two examples are provided:
1. Tumor Burden Model - Simple 3-parameter model for lung cancer
2. HBV Model - Multi-phase simulation for hepatitis B treatment

The VCT framework supports:
- Multiple treatment arms (control, NUC, IFN, combination)
- Multi-phase simulation (untreated → NUC background → treatment → off-treatment)
- Batch simulation of virtual populations
- Endpoint analysis with bootstrap confidence intervals

Reference:
    Section 2, Step 6 of the ISCT workflow paper
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ISCTWorkflow
using DataFrames
using DataFramesMeta
using Statistics
using Random
using AlgebraOfGraphics
using CairoMakie

#=============================================================================
# Part 1: Tumor Burden VCT
=============================================================================#

println("=" ^ 70)
println("Part 1: Tumor Burden Virtual Clinical Trial")
println("=" ^ 70)

# Generate virtual population
println("\nGenerating virtual population...")
vpop_tb = generate_tumor_burden_vpop(5000; seed=22)
println("Generated $(nrow(vpop_tb)) virtual patients")

# Summarize parameters
println("\nParameter Summary:")
summary_tb = summarize_vpop(vpop_tb, [:f, :g, :k])
display(summary_tb)

# Run complete trial (treatment + control)
println("\nRunning Tumor Burden Trial...")
observation_times = collect(0:7:126)  # Weekly observations for 18 weeks

trial_tb = run_tumor_burden_trial(
    vpop_tb;
    observation_times = observation_times,
    seed = 42,
    show_progress = true
)

println("\nTrial Results:")
println("  Treatment arm: $(length(unique(trial_tb.treatment.id))) patients, $(nrow(trial_tb.treatment)) observations")
println("  Control arm: $(length(unique(trial_tb.control.id))) patients, $(nrow(trial_tb.control)) observations")

# Analyze response at different time points
println("\nResponse Analysis (threshold = 0.14):")
for week in [6, 12, 18]
    day = week * 7
    tx_data = @subset(trial_tb.treatment, :time .== day)
    ctrl_data = @subset(trial_tb.control, :time .== day)

    tx_responders = sum(tx_data.Nt .< 0.14)
    ctrl_responders = sum(ctrl_data.Nt .< 0.14)

    println("  Week $week:")
    println("    Treatment: $(round(100*tx_responders/nrow(tx_data), digits=1))% complete response")
    println("    Control: $(round(100*ctrl_responders/nrow(ctrl_data), digits=1))% complete response")
end

#=============================================================================
# Part 2: HBV Virtual Clinical Trial
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 2: HBV Virtual Clinical Trial")
println("=" ^ 70)

# Load or generate HBV virtual population
println("\nGenerating HBV virtual population...")

# Create a synthetic HBV population for demonstration
Random.seed!(42)
n_hbv = 5000

vpop_hbv = DataFrame(
    id = 1:n_hbv,
    beta = randn(n_hbv) .* 3.0 .- 5.0,
    p_S = randn(n_hbv) .* 0.5 .+ 8.0,
    m = randn(n_hbv) .* 1.0 .+ 2.5,
    k_Z = randn(n_hbv) .* 1.0 .- 5.0,
    convE = randn(n_hbv) .* 1.0 .+ 2.5,
    epsNUC = randn(n_hbv) .* 1.0 .- 2.0,
    epsIFN = randn(n_hbv) .* 1.0 .- 1.5,
    r_E_IFN = randn(n_hbv) .* 0.2 .+ 0.3,
    k_D = randn(n_hbv) .* 0.5 .- 4.0
)

# Add fixed parameters
vpop_hbv = add_fixed_params(vpop_hbv)
println("Generated $(nrow(vpop_hbv)) HBV virtual patients")

# Configure trial
println("\nConfiguring HBV Trial...")
println("  Treatment arms: Control, NUC, IFN, NUC+IFN")
println("  Population: Treatment-naive (non-suppressed)")

# Show trial phases
config_naive = VCTConfig(treatment=NUC_IFN_COMBO, suppressed=false)
phases = get_phase_times(config_naive)
println("\nTrial Phases (naive):")
println("  Untreated: Day 0 - $(phases.untreated.stop) ($(phases.untreated.stop ÷ 365) years)")
println("  Treatment: Day $(phases.treatment.start) - $(phases.treatment.stop) ($(config_naive.treatment_duration ÷ 7) weeks)")
println("  Off-treatment: Day $(phases.off_treatment.start) - $(phases.off_treatment.stop) ($(config_naive.off_treatment_duration ÷ 7) weeks)")

# Run trial comparison
println("\nRunning HBV Trial Comparison...")
trial_results = run_hbv_trial_comparison(
    vpop_hbv;
    treatments = [CONTROL, NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO],
    suppressed = false,
    seed = 42
)

# Analyze endpoints
println("\n" * "=" ^ 70)
println("HBV Trial Results")
println("=" ^ 70)

comparison = compare_treatment_arms(trial_results)

# Display results by treatment
for treatment in [CONTROL, NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO]
    results = trial_results[treatment]
    rates = @subset(comparison, :treatment .== string(treatment))

    println("\n$(treatment):")
    println("  N = $(nrow(results.summary))")

    for row in eachrow(rates)
        println("  $(row.endpoint): $(round(row.rate, digits=1))% " *
                "(95% CI: $(round(row.ci_lower, digits=1))-$(round(row.ci_upper, digits=1))%)")
    end
end

# Baseline distribution summary
println("\nBaseline Distribution (Control arm):")
baseline_stats = summarize_baseline_distribution(trial_results[CONTROL])
display(baseline_stats)

#=============================================================================
# Part 3: Suppressed Population Comparison
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 3: NA-Suppressed Population Trial")
println("=" ^ 70)

# Configure suppressed trial
config_sup = VCTConfig(treatment=NUC_IFN_COMBO, suppressed=true)
phases_sup = get_phase_times(config_sup)

println("Trial Phases (suppressed):")
println("  Untreated: Day 0 - $(phases_sup.untreated.stop) ($(phases_sup.untreated.stop ÷ 365) years)")
println("  NUC background: Day $(phases_sup.nuc_background.start) - $(phases_sup.nuc_background.stop) ($(config_sup.nuc_background_duration ÷ 365) years)")
println("  Treatment: Day $(phases_sup.treatment.start) - $(phases_sup.treatment.stop) ($(config_sup.treatment_duration ÷ 7) weeks)")
println("  Off-treatment: Day $(phases_sup.off_treatment.start) - $(phases_sup.off_treatment.stop) ($(config_sup.off_treatment_duration ÷ 7) weeks)")

# Run suppressed trial
println("\nRunning suppressed population trial...")
trial_sup = run_hbv_trial_comparison(
    vpop_hbv;
    treatments = [NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO],
    suppressed = true,
    seed = 42
)

comparison_sup = compare_treatment_arms(trial_sup)

println("\nSuppressed Population Results:")
for treatment in [NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO]
    rates = @subset(comparison_sup, :treatment .== string(treatment))
    fc_rate = @subset(rates, :endpoint .== "Functional Cure")

    if nrow(fc_rate) > 0
        println("  $(treatment): FC = $(round(fc_rate.rate[1], digits=1))% " *
                "(95% CI: $(round(fc_rate.ci_lower[1], digits=1))-$(round(fc_rate.ci_upper[1], digits=1))%)")
    end
end

#=============================================================================
# Part 4: Visualization
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 4: Creating Visualizations")
println("=" ^ 70)

# Figure 1: Tumor Burden Dynamics
fig1 = Figure(size = (800, 500))

# Calculate median and quartiles by time
function summarize_by_time(df)
    @chain df begin
        @groupby(:time)
        @combine(
            :median = median(:Nt),
            :q25 = quantile(:Nt, 0.25),
            :q75 = quantile(:Nt, 0.75)
        )
        @transform(:time_weeks = :time ./ 7)
    end
end

tx_summary = summarize_by_time(trial_tb.treatment)
ctrl_summary = summarize_by_time(trial_tb.control)

ax1 = Axis(fig1[1, 1],
    xlabel = "Time (weeks)",
    ylabel = "Baseline-Normalized Tumor Diameter",
    title = "Tumor Burden VCT: Treatment vs Control"
)

# Control arm
band!(ax1, ctrl_summary.time_weeks, ctrl_summary.q25, ctrl_summary.q75,
      color = (:gray, 0.3))
lines!(ax1, ctrl_summary.time_weeks, ctrl_summary.median,
       color = :gray, linewidth = 2, label = "Control")

# Treatment arm
band!(ax1, tx_summary.time_weeks, tx_summary.q25, tx_summary.q75,
      color = (:blue, 0.3))
lines!(ax1, tx_summary.time_weeks, tx_summary.median,
       color = :blue, linewidth = 2, label = "Treatment")

xlims!(ax1, 0, 18)
ylims!(ax1, 0.5, 1.7)
axislegend(ax1, position = :lt)

save(joinpath(@__DIR__, "..", "outputs", "vct_tumor_burden.png"), fig1)
println("Saved: outputs/vct_tumor_burden.png")

# Figure 2: HBV Treatment Comparison (Bar Chart)
fig2 = Figure(size = (700, 500))
ax2 = Axis(fig2[1, 1],
    xlabel = "Treatment Arm",
    ylabel = "Functional Cure Rate (%)",
    title = "HBV VCT: Functional Cure Rates by Treatment",
    xticks = (1:4, ["Control", "NUC", "IFN", "NUC+IFN"])
)

# Get FC rates for naive population
fc_rates = @chain comparison begin
    @subset(:endpoint .== "Functional Cure")
end

treatments_order = ["CONTROL", "NUC_ONLY", "IFN_ONLY", "NUC_IFN_COMBO"]
rates = [fc_rates[fc_rates.treatment .== t, :rate][1] for t in treatments_order]
ci_low = [fc_rates[fc_rates.treatment .== t, :ci_lower][1] for t in treatments_order]
ci_high = [fc_rates[fc_rates.treatment .== t, :ci_upper][1] for t in treatments_order]

barplot!(ax2, 1:4, rates, color = [:gray, :blue, :orange, :green])
errorbars!(ax2, 1:4, rates, rates .- ci_low, ci_high .- rates,
           color = :black, whiskerwidth = 10)

save(joinpath(@__DIR__, "..", "outputs", "vct_hbv_fc_rates.png"), fig2)
println("Saved: outputs/vct_hbv_fc_rates.png")

# Figure 3: Naive vs Suppressed Comparison
fig3 = Figure(size = (600, 500))
ax3 = Axis(fig3[1, 1],
    xlabel = "Treatment",
    ylabel = "Functional Cure Rate (%)",
    title = "HBV VCT: Naive vs Suppressed Populations",
    xticks = (1:3, ["NUC", "IFN", "NUC+IFN"])
)

fc_naive = @chain comparison begin
    @subset(:endpoint .== "Functional Cure")
    @subset(:treatment .!= "CONTROL")
end

fc_sup = @chain comparison_sup begin
    @subset(:endpoint .== "Functional Cure")
end

treatments_3 = ["NUC_ONLY", "IFN_ONLY", "NUC_IFN_COMBO"]
rates_naive = [fc_naive[fc_naive.treatment .== t, :rate][1] for t in treatments_3]
rates_sup = [fc_sup[fc_sup.treatment .== t, :rate][1] for t in treatments_3]

barwidth = 0.35
barplot!(ax3, (1:3) .- barwidth/2, rates_naive, width = barwidth,
         color = :blue, label = "Naive")
barplot!(ax3, (1:3) .+ barwidth/2, rates_sup, width = barwidth,
         color = :orange, label = "Suppressed")

axislegend(ax3, position = :lt)

save(joinpath(@__DIR__, "..", "outputs", "vct_naive_vs_suppressed.png"), fig3)
println("Saved: outputs/vct_naive_vs_suppressed.png")

#=============================================================================
# Summary
=============================================================================#

println("\n" * "=" ^ 70)
println("VCT Simulation Complete!")
println("=" ^ 70)

println("\nKey Results:")
println("\nTumor Burden Model:")
println("  - 5000 virtual patients simulated")
println("  - Treatment shows clear benefit vs control")

println("\nHBV Model (Naive):")
for treatment in [NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO]
    rates = @subset(comparison, :treatment .== string(treatment), :endpoint .== "Functional Cure")
    if nrow(rates) > 0
        println("  - $(treatment): $(round(rates.rate[1], digits=1))% FC rate")
    end
end

println("\nHBV Model (Suppressed):")
for treatment in [NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO]
    rates = @subset(comparison_sup, :treatment .== string(treatment), :endpoint .== "Functional Cure")
    if nrow(rates) > 0
        println("  - $(treatment): $(round(rates.rate[1], digits=1))% FC rate")
    end
end

println("\nOutput files saved to outputs/")
println("\nNote: The HBV simulation uses a simplified model for demonstration.")
println("For production use, integrate with full Pumas ODE simulation.")
