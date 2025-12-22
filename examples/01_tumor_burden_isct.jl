#=
Tumor Burden In Silico Clinical Trial Example

This script demonstrates the complete ISCT workflow for the tumor burden model
as described in Section 3.1 of the accompanying publication.

Steps:
1. Generate virtual population using Gaussian copulas
2. Simulate treatment and control arms
3. Analyze response rates
4. Visualize results

Reference:
    Qi T, Cao Y. Virtual Clinical Trials: A Tool for Predicting Patients Who May
    Benefit From Treatment Beyond Progression With Pembrolizumab in Non-Small Cell
    Lung Cancer. CPT Pharmacometrics Syst Pharmacol. 2023;12:236-249.
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ISCTWorkflow
using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

#=============================================================================
# Step 1: Generate Virtual Population
=============================================================================#

println("=" ^ 60)
println("Step 1: Generating Virtual Population")
println("=" ^ 60)

# Parameters from Qi & Cao (2023):
# - f: Treatment-sensitive fraction, median=0.27, ω=2.16, range [0,1]
# - g: Growth rate, median=0.0013 d⁻¹, ω=1.57, range [0, 0.13]
# - k: Death rate, median=0.0091 d⁻¹, ω=1.24, range [0, 1.6]
# - Correlation: r(f,g) = -0.64

n_patients = 10_000
vpop = generate_tumor_burden_vpop(n_patients; seed=22)

# Display summary statistics
param_names = [:f, :g, :k]
summary_stats = summarize_vpop(vpop, param_names)
println("\nVirtual Population Summary:")
println(summary_stats)

# Validate correlations
expected_corr = TumorBurdenCorrelation()
corr_validation = validate_correlations(vpop, param_names, expected_corr)
println("\nCorrelation Validation:")
println("  Expected r(f,g) = -0.64")
println("  Observed correlation matrix:")
display(round.(corr_validation.observed, digits=3))
println("  Max difference: ", round(corr_validation.max_difference, digits=4))
println("  Valid (< 5% tolerance): ", corr_validation.valid)

#=============================================================================
# Step 2: Run Virtual Clinical Trial
=============================================================================#

println("\n" * "=" ^ 60)
println("Step 2: Running Virtual Clinical Trial")
println("=" ^ 60)

# Observation times: 0 to 126 days (18 weeks)
observation_times = 0:1:126

# Run complete VCT (both arms)
# Note: simulate_vpop now uses tumor_burden_model internally with computed random effects
println("Simulating treatment arm...")
treatment_results = simulate_vpop(
    vpop,
    collect(observation_times);
    treatment = 1
)

println("Simulating control arm...")
control_results = simulate_vpop(
    vpop,
    collect(observation_times);
    treatment = 0
)

println("  Treatment arm: $(nrow(treatment_results)) observations")
println("  Control arm: $(nrow(control_results)) observations")

#=============================================================================
# Step 3: Analyze Response Rates
=============================================================================#

println("\n" * "=" ^ 60)
println("Step 3: Analyzing Response Rates")
println("=" ^ 60)

# Threshold for complete response (from paper)
# Minimum detectable PET is ~7mm, tumor baseline 3-5cm for Stage IIIA
threshold = 0.7 / 5  # = 0.14

# Analyze response rates at 6, 12, 18 weeks
response_analysis = analyze_response_rates(
    treatment_results;
    time_weeks = [6, 12, 18],
    threshold = threshold,
    n_bootstrap = 100,
    bootstrap_size = 1000
)

println("\nComplete Response Rates (%):")
println(response_analysis)

# Calculate treatment effect at final time (18 weeks)
final_time = 126
treatment_final = @chain treatment_results begin
    @subset(:time .== final_time)
    @select(:id, :Nt)
end
control_final = @chain control_results begin
    @subset(:time .== final_time)
    @select(:id, :Nt)
end

# Merge and calculate percent reduction
combined = innerjoin(
    rename(treatment_final, :Nt => :Nt_treatment),
    rename(control_final, :Nt => :Nt_control),
    on = :id
)
combined.percent_reduction = 100 .* (combined.Nt_control .- combined.Nt_treatment) ./ combined.Nt_control

println("\nTreatment Effect at Week 18:")
println("  Median tumor size reduction: $(round(median(combined.percent_reduction), digits=1))%")
println("  5th percentile: $(round(quantile(combined.percent_reduction, 0.05), digits=1))%")
println("  95th percentile: $(round(quantile(combined.percent_reduction, 0.95), digits=1))%")

#=============================================================================
# Step 4: Visualization
=============================================================================#

println("\n" * "=" ^ 60)
println("Step 4: Creating Visualizations")
println("=" ^ 60)

# Prepare data for plotting
# Calculate median and quartiles at each time point
function summarize_dynamics(results::DataFrame)
    @chain results begin
        @groupby(:time)
        @combine(
            :median = median(:Nt),
            :q25 = quantile(:Nt, 0.25),
            :q75 = quantile(:Nt, 0.75)
        )
        @transform(:time_weeks = :time ./ 7)
    end
end

treatment_summary = summarize_dynamics(treatment_results)
treatment_summary.arm .= "Treatment"

control_summary = summarize_dynamics(control_results)
control_summary.arm .= "Control"

dynamics_df = vcat(treatment_summary, control_summary)

# Plot 1: Tumor dynamics comparison
fig1 = Figure(size = (800, 500))
ax1 = Axis(
    fig1[1, 1],
    xlabel = "Time (weeks)",
    ylabel = "Baseline-Normalized Tumor Diameter",
    title = "Virtual Clinical Trial: Tumor Burden Dynamics"
)

# Plot quartile bands and medians
for (arm, color) in [("Control", :gray60), ("Treatment", :lightblue)]
    arm_data = filter(row -> row.arm == arm, dynamics_df)

    # Quartile band
    band!(
        ax1,
        arm_data.time_weeks,
        arm_data.q25,
        arm_data.q75,
        color = (color, 0.3)
    )

    # Median line
    lines!(
        ax1,
        arm_data.time_weeks,
        arm_data.median,
        color = color,
        linewidth = 2,
        label = arm
    )
end

xlims!(ax1, 0, 18)
ylims!(ax1, 0.5, 1.7)
axislegend(ax1, position = :lt)
fig1
save(joinpath(@__DIR__, "..", "outputs", "tumor_dynamics.png"), fig1)
println("Saved: outputs/tumor_dynamics.png")

# Plot 2: Response rates bar chart
fig2 = Figure(size = (600, 400))
ax2 = Axis(
    fig2[1, 1],
    xlabel = "Patients with Complete Response (%)",
    ylabel = "Weeks of Treatment",
    title = "Complete Response Rates by Treatment Duration",
    yticks = [6, 12, 18]
)

# Horizontal bar plot with error bars
barplot!(
    ax2,
    response_analysis.week,
    response_analysis.median_rate,
    direction = :x,
    color = (:gray50, 0.5),
    strokewidth = 1,
    strokecolor = :gray30
)
fig2
# Error bars
errorbars!(
    ax2,
    response_analysis.median_rate,
    response_analysis.week,
    response_analysis.median_rate .- response_analysis.ci_lower,
    response_analysis.ci_upper .- response_analysis.median_rate,
    direction = :x,
    color = :gray30,
    whiskerwidth = 10
)

ylims!(ax2, 3, 21)

save(joinpath(@__DIR__, "..", "outputs", "response_rates.png"), fig2)
println("Saved: outputs/response_rates.png")

# Plot 3: Virtual population parameter distributions
fig3 = Figure(size = (900, 300))

# f distribution
ax3a = Axis(fig3[1, 1], xlabel = "f (Treatment-sensitive fraction)", ylabel = "Density")
hist!(ax3a, vpop.f, bins = 50, color = (:blue, 0.5), normalization = :pdf)
vlines!(ax3a, [0.27], color = :red, linestyle = :dash, label = "Median (0.27)")

# g distribution
ax3b = Axis(fig3[1, 2], xlabel = "g (Growth rate, d⁻¹)", ylabel = "Density")
hist!(ax3b, vpop.g, bins = 50, color = (:green, 0.5), normalization = :pdf)
vlines!(ax3b, [0.0013], color = :red, linestyle = :dash, label = "Median (0.0013)")

# k distribution
ax3c = Axis(fig3[1, 3], xlabel = "k (Death rate, d⁻¹)", ylabel = "Density")
hist!(ax3c, vpop.k, bins = 50, color = (:orange, 0.5), normalization = :pdf)
vlines!(ax3c, [0.0091], color = :red, linestyle = :dash, label = "Median (0.0091)")

Label(fig3[0, :], "Virtual Population Parameter Distributions", fontsize = 16)
fig3
save(joinpath(@__DIR__, "..", "outputs", "vpop_distributions.png"), fig3)
println("Saved: outputs/vpop_distributions.png")

# Plot 4: Parameter correlations (f vs g)
fig4 = Figure(size = (500, 500))
ax4 = Axis(
    fig4[1, 1],
    xlabel = "f (Treatment-sensitive fraction)",
    ylabel = "g (Growth rate, d⁻¹)",
    title = "Parameter Correlation: f vs g (expected r = -0.64)"
)

# Subsample for visualization (plotting 10,000 points is slow)
n_plot = min(1000, n_patients)
scatter!(
    ax4,
    vpop.f[1:n_plot],
    vpop.g[1:n_plot],
    color = (:blue, 0.3),
    markersize = 4
)
fig4
save(joinpath(@__DIR__, "..", "outputs", "parameter_correlation.png"), fig4)
println("Saved: outputs/parameter_correlation.png")

println("\n" * "=" ^ 60)
println("ISCT Complete!")
println("=" ^ 60)
println("\nKey Results:")
println("  - Generated $(n_patients) virtual patients")
println("  - Correlation validation: $(corr_validation.valid ? "PASSED" : "FAILED")")
println("  - Complete response rate at 18 weeks: $(round(response_analysis.median_rate[3], digits=2))%")
println("  - Median tumor size reduction: $(round(median(combined.percent_reduction), digits=1))%")
