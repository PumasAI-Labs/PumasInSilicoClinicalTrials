#=
Visualization Example

This script demonstrates the comprehensive visualization capabilities
of the ISCT workflow package using AlgebraOfGraphics.jl and CairoMakie.

Visualizations include:
1. Virtual population distributions and correlations
2. VCT simulation dynamics (treatment vs control)
3. Response waterfall plots
4. Treatment comparison bar charts
5. MILP calibration results
6. GSA sensitivity indices

Reference:
    Supporting visualization for the 7-step ISCT workflow
=#
include("../src/ISCTWorkflow.jl")
using .ISCTWorkflow
using DataFrames
using DataFramesMeta
using Statistics
using Random
using CairoMakie

# Set the ISCT theme
set_isct_theme!()

#=============================================================================
# Part 1: Virtual Population Visualization
=============================================================================#

println("=" ^ 70)
println("Part 1: Virtual Population Visualization")
println("=" ^ 70)

# Generate virtual population
println("\nGenerating virtual population...")
vpop = generate_tumor_burden_vpop(5000; seed=22)
println("Generated $(nrow(vpop)) virtual patients")

# Plot parameter distributions
println("\nCreating parameter distribution plots...")
fig1 = plot_parameter_distributions(vpop, [:f, :g, :k];
    title = "Tumor Burden Model Parameters"
)
save_figure(fig1, joinpath(@__DIR__, "..", "outputs", "viz_param_distributions.png"))

# Plot parameter correlations
println("Creating correlation matrix...")
fig2 = plot_parameter_correlations(vpop, [:f, :g, :k];
    title = "Parameter Correlations (Tumor Burden)"
)
save_figure(fig2, joinpath(@__DIR__, "..", "outputs", "viz_param_correlations.png"))

# Summary statistics
println("\nParameter Summary:")
for param in [:f, :g, :k]
    vals = vpop[!, param]
    println("  $param: mean=$(round(mean(vals), digits=4)), " *
            "std=$(round(std(vals), digits=4)), " *
            "range=[$(round(minimum(vals), digits=4)), $(round(maximum(vals), digits=4))]")
end

#=============================================================================
# Part 2: VCT Dynamics Visualization
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 2: VCT Dynamics Visualization")
println("=" ^ 70)

# Run tumor burden VCT
println("\nRunning tumor burden VCT simulation...")
observation_times = collect(0:7:126)  # Weekly for 18 weeks

trial_results = run_tumor_burden_trial(
    vpop;
    observation_times = observation_times,
    seed = 42,
    show_progress = true
)

println("  Treatment arm: $(length(unique(trial_results.treatment.id))) patients")
println("  Control arm: $(length(unique(trial_results.control.id))) patients")

# Plot tumor dynamics
println("\nCreating tumor dynamics plot...")
fig3 = plot_tumor_dynamics(
    trial_results.treatment,
    trial_results.control;
    title = "Tumor Burden: Treatment vs Control"
)
save_figure(fig3, joinpath(@__DIR__, "..", "outputs", "viz_tumor_dynamics.png"))

# Create waterfall plot
println("Creating waterfall plot...")
fig4 = plot_response_waterfall(
    trial_results.treatment,
    0,      # baseline time
    126;    # final time (18 weeks)
    threshold = 0.14,
    title = "Treatment Response at Week 18",
    max_patients = 100
)
save_figure(fig4, joinpath(@__DIR__, "..", "outputs", "viz_waterfall.png"))

#=============================================================================
# Part 3: Treatment Comparison
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 3: Treatment Comparison Visualization")
println("=" ^ 70)

# Calculate response rates at different time points
println("\nCalculating response rates...")

function calculate_response_stats(df, time_days, threshold=0.14)
    subset = @subset(df, :time .== time_days)
    rate = 100 * sum(subset.Nt .< threshold) / nrow(subset)
    # Simple bootstrap for CI
    n_boot = 100
    rates = Float64[]
    for _ in 1:n_boot
        sample_idx = rand(1:nrow(subset), nrow(subset))
        boot_rate = 100 * sum(subset.Nt[sample_idx] .< threshold) / nrow(subset)
        push!(rates, boot_rate)
    end
    return (rate, quantile(rates, 0.025), quantile(rates, 0.975))
end

# Response at multiple time points
response_data = DataFrame()

for week in [6, 12, 18]
    day = week * 7
    tx_stats = calculate_response_stats(trial_results.treatment, day)
    ctrl_stats = calculate_response_stats(trial_results.control, day)

    push!(response_data, (
        week = week,
        treatment = "Treatment",
        rate = tx_stats[1],
        ci_lower = tx_stats[2],
        ci_upper = tx_stats[3]
    ))
    push!(response_data, (
        week = week,
        treatment = "Control",
        rate = ctrl_stats[1],
        ci_lower = ctrl_stats[2],
        ci_upper = ctrl_stats[3]
    ))
end

println("\nResponse Rates:")
display(response_data)

# Create grouped bar chart
println("\nCreating response comparison plot...")
fig5 = Figure(size = (700, 450))

ax5 = Axis(fig5[1, 1],
    xlabel = "Week",
    ylabel = "Complete Response Rate (%)",
    title = "Response Rate Over Time",
    xticks = ([1, 2, 3], ["Week 6", "Week 12", "Week 18"])
)

# Separate treatment and control data
tx_data = @subset(response_data, :treatment .== "Treatment")
ctrl_data = @subset(response_data, :treatment .== "Control")

barwidth = 0.35
x_tx = [1, 2, 3] .- barwidth/2
x_ctrl = [1, 2, 3] .+ barwidth/2

barplot!(ax5, x_ctrl, ctrl_data.rate, width=barwidth, color=:gray, label="Control")
errorbars!(ax5, x_ctrl, ctrl_data.rate,
           ctrl_data.rate .- ctrl_data.ci_lower, ctrl_data.ci_upper .- ctrl_data.rate,
           color=:black, whiskerwidth=8)

barplot!(ax5, x_tx, tx_data.rate, width=barwidth, color=:blue, label="Treatment")
errorbars!(ax5, x_tx, tx_data.rate,
           tx_data.rate .- tx_data.ci_lower, tx_data.ci_upper .- tx_data.rate,
           color=:black, whiskerwidth=8)

axislegend(ax5, position=:lt)

save_figure(fig5, joinpath(@__DIR__, "..", "outputs", "viz_response_comparison.png"))

#=============================================================================
# Part 4: Calibration Visualization
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 4: Calibration Visualization")
println("=" ^ 70)

# Create synthetic uncalibrated population
println("\nCreating synthetic populations for calibration demo...")
Random.seed!(42)
n_orig = 10000
vpop_orig = DataFrame(
    id = 1:n_orig,
    HBsAg = randn(n_orig) .* 2.5 .+ 3.0  # Wide distribution
)

# Define target
target_edges = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
target_pcts = [5.0, 15.0, 30.0, 30.0, 15.0, 5.0]

target = create_target_from_specification(target_edges, target_pcts; variable_name=:HBsAg)

# Run calibration
println("Running MILP calibration...")
result = solve_milp_calibration(vpop_orig.HBsAg, target, 0.15; time_limit=30.0)
print_calibration_summary(result)

# Create calibrated population
vpop_calib = vpop_orig[result.selected_indices, :]

# Plot comparison
println("\nCreating calibration comparison plot...")
fig6 = plot_vpop_comparison(vpop_orig, vpop_calib, :HBsAg;
    title = "MILP Calibration: Before vs After"
)
save_figure(fig6, joinpath(@__DIR__, "..", "outputs", "viz_calibration_comparison.png"))

# Create bin comparison
println("Creating bin distribution comparison...")
fig7 = Figure(size = (700, 500))

# Calculate percentages in each bin
function bin_percentages(values, edges)
    counts = zeros(length(edges) - 1)
    for v in values
        for i in 1:length(counts)
            if v >= edges[i] && v < edges[i+1]
                counts[i] += 1
                break
            end
        end
    end
    return 100 .* counts ./ length(values)
end

pct_orig = bin_percentages(vpop_orig.HBsAg, target_edges)
pct_calib = bin_percentages(vpop_calib.HBsAg, target_edges)
pct_target = target_pcts

bin_labels = ["1-2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-5"]
n_bins = length(bin_labels)

ax7 = Axis(fig7[1, 1],
    xlabel = "HBsAg Range (log₁₀ IU/mL)",
    ylabel = "Percentage (%)",
    title = "Distribution Matching",
    xticks = (1:n_bins, bin_labels)
)

barwidth7 = 0.25
barplot!(ax7, (1:n_bins) .- barwidth7, pct_orig, width=barwidth7, color=:blue, label="Original")
barplot!(ax7, (1:n_bins), pct_calib, width=barwidth7, color=:green, label="Calibrated")
barplot!(ax7, (1:n_bins) .+ barwidth7, pct_target, width=barwidth7, color=:orange, label="Target")

axislegend(ax7, position=:rt)

save_figure(fig7, joinpath(@__DIR__, "..", "outputs", "viz_bin_comparison.png"))

#=============================================================================
# Part 5: GSA Visualization
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 5: GSA Visualization")
println("=" ^ 70)

# Run GSA
println("\nRunning eFAST analysis...")
gsa_result = run_tumor_burden_gsa(
    method = :efast,
    n_samples = 300,
    treatment = 1
)

print_gsa_summary(gsa_result)

# Plot sensitivity indices
println("\nCreating GSA bar chart...")
fig8 = plot_gsa_indices(gsa_result, :final_tumor;
    title = "Tumor Burden GSA"
)
save_figure(fig8, joinpath(@__DIR__, "..", "outputs", "viz_gsa_indices.png"))

# Create GSA heatmap
println("Creating GSA heatmap...")
fig9 = plot_gsa_heatmap(gsa_result;
    index_type = :total_order,
    title = "Total Order Sensitivity Heatmap"
)
save_figure(fig9, joinpath(@__DIR__, "..", "outputs", "viz_gsa_heatmap.png"))

#=============================================================================
# Part 6: Summary Dashboard
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 6: Summary Dashboard")
println("=" ^ 70)

println("\nCreating ISCT summary figure...")
fig10 = create_isct_summary_figure(
    vpop,
    [:f, :g, :k],
    trial_results.treatment,
    trial_results.control;
    title = "ISCT Workflow Summary: Tumor Burden Model"
)
save_figure(fig10, joinpath(@__DIR__, "..", "outputs", "viz_isct_summary.png"))

#=============================================================================
# Part 7: Quick Exploration Utilities
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 7: Quick Exploration Utilities Demo")
println("=" ^ 70)

# Quick histogram
println("\nQuick histogram example...")
fig_quick1 = quick_hist(vpop.f; title="Treatment-Sensitive Fraction (f)", bins=40)
save_figure(fig_quick1, joinpath(@__DIR__, "..", "outputs", "viz_quick_hist.png"))

# Quick scatter
println("Quick scatter example...")
fig_quick2 = quick_scatter(vpop.g, vpop.f;
    xlabel="Growth Rate (g)", ylabel="Sensitive Fraction (f)",
    title="g vs f Correlation"
)
save_figure(fig_quick2, joinpath(@__DIR__, "..", "outputs", "viz_quick_scatter.png"))

#=============================================================================
# Summary
=============================================================================#

println("\n" * "=" ^ 70)
println("Visualization Example Complete!")
println("=" ^ 70)

println("\nGenerated Figures:")
println("  1. viz_param_distributions.png - Parameter histograms")
println("  2. viz_param_correlations.png - Correlation matrix")
println("  3. viz_tumor_dynamics.png - Treatment vs control dynamics")
println("  4. viz_waterfall.png - Response waterfall plot")
println("  5. viz_response_comparison.png - Response rate comparison")
println("  6. viz_calibration_comparison.png - Before/after calibration")
println("  7. viz_bin_comparison.png - Distribution matching")
println("  8. viz_gsa_indices.png - GSA sensitivity bar chart")
println("  9. viz_gsa_heatmap.png - GSA sensitivity heatmap")
println(" 10. viz_isct_summary.png - ISCT workflow summary dashboard")
println(" 11. viz_quick_hist.png - Quick histogram utility")
println(" 12. viz_quick_scatter.png - Quick scatter utility")

println("\nVisualization Capabilities:")
println("  - Virtual population: distributions, correlations, comparisons")
println("  - VCT results: dynamics with CI bands, waterfall, treatment comparison")
println("  - Calibration: before/after, bin matching, Pareto fronts")
println("  - GSA: sensitivity indices, heatmaps, treatment comparisons")
println("  - Utilities: quick_hist, quick_scatter for exploration")

println("\nAll figures saved to outputs/")
