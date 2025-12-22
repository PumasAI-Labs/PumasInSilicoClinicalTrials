#=
HBV Population Dynamics Visualization Example

This script demonstrates population-level mean ± CI band plots for HBV infection
dynamics, showing:
1. Natural history (untreated) with acute vs chronic infection outcomes
2. Treatment response dynamics for different treatment arms
3. All four biomarkers: HBsAg, Viral Load, ALT, Effector T cells

The visualization follows Figure 1 from the CPT 2025 paper, displaying:
- Population median as center line
- 90% CI bands (q05 to q95)
- IQR bands (q25 to q75)
- LOQ threshold lines for HBsAg and viral load

Reference:
    "A Step-by-Step Workflow for Performing In Silico Clinical Trials
    With Nonlinear Mixed Effects Models" - CPT: Pharmacometrics & Systems
    Pharmacology (2025)
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ISCTWorkflow
using DataFrames
using DataFramesMeta
using Statistics
using Random
using CairoMakie
using LinearAlgebra

# Set the ISCT theme
set_isct_theme!()

# Create output directory
output_dir = joinpath(@__DIR__, "..", "outputs")
mkpath(output_dir)

#=============================================================================
# Part 1: Load or Generate Virtual Population
=============================================================================#

println("=" ^ 70)
println("Part 1: Loading HBV Virtual Population")
println("=" ^ 70)

# Try to load from CSV, otherwise create synthetic demo population
param_csv = joinpath(@__DIR__, "..", "..", "ISCT SoC and MBMA comparison", "Parameters_Naive.csv")

if isfile(param_csv)
    println("Loading from: $param_csv")
    vpop_full = load_hbv_vpop(param_csv)
    println("Loaded $(nrow(vpop_full)) virtual patients")
else
    println("Parameter CSV not found. Creating synthetic demo population...")

    # Create demo population with approximate HBV parameter distributions
    demo_specs = [
        HBVParameterSpec(:beta, -5.0, 3.0),
        HBVParameterSpec(:p_S, 8.0, 0.5),
        HBVParameterSpec(:m, 2.5, 1.0),
        HBVParameterSpec(:k_Z, -5.0, 1.0),
        HBVParameterSpec(:convE, 2.5, 1.0),
        HBVParameterSpec(:epsNUC, -2.0, 1.0),
        HBVParameterSpec(:epsIFN, -1.5, 1.0),
        HBVParameterSpec(:r_E_IFN, 0.3, 0.2),
        HBVParameterSpec(:k_D, -4.0, 0.5)
    ]

    # Create correlation matrix with key correlations
    demo_corr = Matrix{Float64}(I, 9, 9)
    # Add some correlations between parameters
    demo_corr[1, 2] = demo_corr[2, 1] = 0.3  # beta-p_S
    demo_corr[6, 7] = demo_corr[7, 6] = 0.5  # epsNUC-epsIFN

    vpop_full = generate_hbv_vpop(5000, demo_specs, demo_corr; seed=42)
    println("Generated $(nrow(vpop_full)) synthetic virtual patients")
end

# Subsample for computational efficiency in this demo
# (Full simulations would use more patients)
n_subsample = 200
vpop = subsample_hbv_vpop(vpop_full, n_subsample; seed=42)
println("Subsampled $(nrow(vpop)) patients for demonstration")

#=============================================================================
# Part 2: Simulate Natural History (Untreated, ~300 days)
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 2: Simulating Natural History (Untreated Phase)")
println("=" ^ 70)

println("\nSimulating untreated HBV infection dynamics...")
println("This simulates ~300 days of natural infection to classify acute vs chronic")

# Simulate natural history
# Note: Without the full Pumas model, this uses placeholder dynamics
# In practice, you would pass model=hbv_model
natural_history_df = simulate_hbv_natural_history(
    vpop;
    duration_days = 300,
    observation_interval = 7,  # Weekly observations
    seed = 42
)

println("Simulation complete: $(length(unique(natural_history_df.id))) patients")
println("Time range: $(minimum(natural_history_df.time)) to $(maximum(natural_history_df.time)) days")

#=============================================================================
# Part 3: Classify Acute vs Chronic Outcomes
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 3: Classifying Patient Outcomes")
println("=" ^ 70)

# Classify outcomes based on clearance during natural history
# Acute: Both HBsAg and viral load drop below LOQ within 300 days
# Chronic: Infection persists
classified_df = classify_hbv_outcome(
    natural_history_df;
    clearance_time = 300,
    hbsag_threshold = log10(0.05),   # LOQ for HBsAg
    viral_threshold = log10(25.0)     # LOQ for viral load
)

# Summary of outcomes
outcome_counts = combine(groupby(classified_df, :outcome), nrow => :n_timepoints)
patient_outcomes = unique(classified_df[:, [:id, :outcome]])
acute_count = sum(patient_outcomes.outcome .== :acute)
chronic_count = sum(patient_outcomes.outcome .== :chronic)

println("\nOutcome Classification:")
println("  Acute (cleared): $acute_count patients ($(round(100*acute_count/nrow(patient_outcomes), digits=1))%)")
println("  Chronic (persistent): $chronic_count patients ($(round(100*chronic_count/nrow(patient_outcomes), digits=1))%)")

#=============================================================================
# Part 4: Plot Natural History with Acute vs Chronic Outcomes
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 4: Visualizing Natural History Dynamics")
println("=" ^ 70)

# Create 2x2 panel plot showing all four biomarkers
println("\nCreating population dynamics plot...")
fig1 = plot_hbv_natural_history(
    classified_df;
    biomarkers = [:log_HBsAg, :log_V, :log_ALT, :log_E],
    title = "HBV Natural History: Acute vs Chronic Infection"
)
save_figure(fig1, joinpath(output_dir, "hbv_natural_history.png"))
println("Saved: outputs/hbv_natural_history.png")

# Create individual biomarker panels with more detail
println("\nCreating individual biomarker panels...")

# HBsAg panel with individual trajectories
fig_hbsag = plot_hbv_biomarker_panel(
    classified_df,
    :log_HBsAg;
    stratify_by = :outcome,
    time_unit = :days,
    show_individual = true,
    n_individual = 30,
    title = "HBsAg Dynamics During Natural History"
)
save_figure(fig_hbsag, joinpath(output_dir, "hbv_hbsag_panel.png"))
println("Saved: outputs/hbv_hbsag_panel.png")

# Viral load panel
fig_viral = plot_hbv_biomarker_panel(
    classified_df,
    :log_V;
    stratify_by = :outcome,
    time_unit = :days,
    show_individual = true,
    n_individual = 30,
    title = "Viral Load Dynamics During Natural History"
)
save_figure(fig_viral, joinpath(output_dir, "hbv_viral_panel.png"))
println("Saved: outputs/hbv_viral_panel.png")

#=============================================================================
# Part 5: Compute Summary Statistics
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 5: Summary Statistics by Time and Outcome")
println("=" ^ 70)

# Summarize each biomarker
for biomarker in [:log_HBsAg, :log_V, :log_ALT, :log_E]
    summary = summarize_hbv_dynamics_by_time(classified_df, biomarker; group_col=:outcome)

    println("\n$(biomarker) Summary (selected time points):")

    # Show baseline (day 0) and final (day 300) for each outcome
    for outcome in [:acute, :chronic]
        outcome_summary = @subset(summary, :outcome .== outcome)
        if nrow(outcome_summary) > 0
            baseline = @subset(outcome_summary, :time .== 0)
            final = @subset(outcome_summary, :time .== maximum(:time))

            if nrow(baseline) > 0 && nrow(final) > 0
                println("  $outcome:")
                println("    Day 0:   median=$(round(baseline.median[1], digits=2)), " *
                        "90% CI=[$(round(baseline.q05[1], digits=2)), $(round(baseline.q95[1], digits=2))]")
                println("    Day $(Int(final.time[1])): median=$(round(final.median[1], digits=2)), " *
                        "90% CI=[$(round(final.q05[1], digits=2)), $(round(final.q95[1], digits=2))]")
            end
        end
    end
end

#=============================================================================
# Part 6: Simulate Treatment Response Dynamics
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 6: Simulating Treatment Response Dynamics")
println("=" ^ 70)

# Use a subset of chronic patients for treatment simulation
chronic_ids = patient_outcomes[patient_outcomes.outcome .== :chronic, :id]
if length(chronic_ids) > 50
    chronic_vpop = @subset(vpop, :id .∈ Ref(chronic_ids[1:50]))
else
    chronic_vpop = @subset(vpop, :id .∈ Ref(chronic_ids))
end

println("Using $(nrow(chronic_vpop)) chronic patients for treatment simulation")

# Create VCT config for treatment
println("\nTreatment phases:")
println("  1. Untreated: $(HBV_PHASES.untreated ÷ 365) years")
println("  2. NUC background: $(HBV_PHASES.nuc_background ÷ 365) years (suppressed only)")
println("  3. Treatment: $(HBV_PHASES.treatment ÷ 7) weeks")
println("  4. Off-treatment: $(HBV_PHASES.off_treatment ÷ 7) weeks")

# Simulate combo treatment
config_combo = VCTConfig(
    treatment = NUC_IFN_COMBO,
    suppressed = true,
    untreated_duration = 365,      # 1 year for visualization
    nuc_background_duration = 365, # 1 year
    treatment_duration = HBV_PHASES.treatment,
    off_treatment_duration = HBV_PHASES.off_treatment,
    observation_interval = 7
)

println("\nSimulating NUC+IFN combination treatment...")
treatment_df = simulate_hbv_dynamics(
    chronic_vpop,
    config_combo;
    seed = 123
)

println("Treatment simulation complete: $(length(unique(treatment_df.id))) patients")

# Get phase times for plotting
phase_times = get_phase_times(config_combo)
println("\nPhase boundaries:")
println("  End untreated: day $(phase_times.untreated.stop)")
println("  End NUC background: day $(phase_times.nuc_background.stop)")
println("  End treatment: day $(phase_times.treatment.stop)")
println("  End off-treatment: day $(phase_times.off_treatment.stop)")

#=============================================================================
# Part 7: Plot Treatment Response
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 7: Visualizing Treatment Response")
println("=" ^ 70)

# Plot treatment response with phase markers
println("\nCreating treatment response plot...")
fig2 = plot_hbv_treatment_response(
    treatment_df;
    biomarkers = [:log_HBsAg, :log_V],
    show_phases = true,
    phase_times = phase_times,
    title = "HBV Treatment Response: NUC+IFN Combination"
)
save_figure(fig2, joinpath(output_dir, "hbv_treatment_response.png"))
println("Saved: outputs/hbv_treatment_response.png")

# Create comprehensive 4-panel treatment dynamics
fig3 = plot_hbv_population_dynamics(
    treatment_df;
    biomarkers = [:log_HBsAg, :log_V, :log_ALT, :log_E],
    stratify_by = nothing,  # No stratification, all patients on treatment
    show_loq = true,
    time_unit = :days,
    title = "Full HBV Treatment Dynamics: All Biomarkers"
)
save_figure(fig3, joinpath(output_dir, "hbv_treatment_all_biomarkers.png"))
println("Saved: outputs/hbv_treatment_all_biomarkers.png")

#=============================================================================
# Part 8: Compare Treatment Arms (Optional)
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 8: Treatment Arm Comparison")
println("=" ^ 70)

# Run VCT for different treatment arms
println("\nRunning VCT for multiple treatment arms...")

# Use smaller subsample for multi-arm comparison
compare_vpop = chronic_vpop[1:min(30, nrow(chronic_vpop)), :]

arm_results = Dict{TreatmentArm, DataFrame}()

for arm in [CONTROL, NUC_ONLY, IFN_ONLY, NUC_IFN_COMBO]
    println("  Simulating $(arm)...")
    config = VCTConfig(
        treatment = arm,
        suppressed = true,
        untreated_duration = 365,
        nuc_background_duration = 365,
        treatment_duration = HBV_PHASES.treatment,
        off_treatment_duration = HBV_PHASES.off_treatment,
        observation_interval = 14  # Bi-weekly for efficiency
    )

    arm_df = simulate_hbv_dynamics(compare_vpop, config; seed=42)
    arm_df.treatment .= string(arm)
    arm_results[arm] = arm_df
end

# Combine all arms
all_arms_df = vcat(values(arm_results)...)
println("\nCombined $(length(unique(all_arms_df.id))) patients across $(length(arm_results)) treatment arms")

# Create comparison figure
println("\nCreating treatment comparison figure...")
fig4 = Figure(size = (1000, 800))

# HBsAg comparison
ax1 = Axis(fig4[1, 1],
    xlabel = "Time (days)",
    ylabel = HBV_BIOMARKER_LABELS[:log_HBsAg],
    title = "HBsAg by Treatment Arm"
)

# Viral load comparison
ax2 = Axis(fig4[1, 2],
    xlabel = "Time (days)",
    ylabel = HBV_BIOMARKER_LABELS[:log_V],
    title = "Viral Load by Treatment Arm"
)

arm_colors = Dict(
    "CONTROL" => colorant"#7f7f7f",
    "NUC_ONLY" => colorant"#1f77b4",
    "IFN_ONLY" => colorant"#ff7f0e",
    "NUC_IFN_COMBO" => colorant"#2ca02c"
)

for (arm_name, arm_df) in pairs(arm_results)
    arm_str = string(arm_name)
    color = arm_colors[arm_str]

    for (ax, biomarker) in [(ax1, :log_HBsAg), (ax2, :log_V)]
        summary = @chain arm_df begin
            @groupby(:time)
            @combine(
                :median = median(cols(biomarker)),
                :q25 = quantile(cols(biomarker), 0.25),
                :q75 = quantile(cols(biomarker), 0.75)
            )
            @orderby(:time)
        end

        band!(ax, summary.time, summary.q25, summary.q75, color = (color, 0.2))
        lines!(ax, summary.time, summary.median, color = color, linewidth = 2, label = arm_str)
    end
end

# Add LOQ lines
hlines!(ax1, [HBV_LOQ_THRESHOLDS[:log_HBsAg]], color = :red, linestyle = :dash, linewidth = 1)
hlines!(ax2, [HBV_LOQ_THRESHOLDS[:log_V]], color = :red, linestyle = :dash, linewidth = 1)

# Add legends
Legend(fig4[2, :], ax1, "Treatment Arm", orientation = :horizontal, tellheight = true)

Label(fig4[0, :], "Treatment Arm Comparison", fontsize = 16)

save_figure(fig4, joinpath(output_dir, "hbv_treatment_comparison.png"))
println("Saved: outputs/hbv_treatment_comparison.png")

#=============================================================================
# Summary
=============================================================================#

println("\n" * "=" ^ 70)
println("HBV Population Dynamics Visualization Complete!")
println("=" ^ 70)

println("\nGenerated Figures:")
println("  1. hbv_natural_history.png - 2x2 panel: All biomarkers, acute vs chronic")
println("  2. hbv_hbsag_panel.png - Single panel: HBsAg with individual trajectories")
println("  3. hbv_viral_panel.png - Single panel: Viral load with individual trajectories")
println("  4. hbv_treatment_response.png - Treatment response (HBsAg, Viral)")
println("  5. hbv_treatment_all_biomarkers.png - 2x2 panel: All biomarkers during treatment")
println("  6. hbv_treatment_comparison.png - Multi-arm treatment comparison")

println("\nVisualization Features:")
println("  - Population median + 90% CI + IQR bands")
println("  - Stratification by outcome (acute/chronic) or treatment arm")
println("  - LOQ threshold lines (HBsAg: $(round(HBV_LOQ_THRESHOLDS[:log_HBsAg], digits=2)), " *
        "Viral: $(round(HBV_LOQ_THRESHOLDS[:log_V], digits=2)))")
println("  - Treatment phase markers")
println("  - Individual patient trajectories option")

println("\nClinical Endpoints:")
println("  - HBsAg Loss: HBsAg < $(LOQ_HBsAg) IU/mL")
println("  - Functional Cure: HBsAg + Viral Load both < LOQ for ≥24 weeks")

println("\nNote: These plots use placeholder dynamics for demonstration.")
println("For production use, pass model=hbv_model to simulation functions.")

println("\nAll figures saved to: $(output_dir)")
