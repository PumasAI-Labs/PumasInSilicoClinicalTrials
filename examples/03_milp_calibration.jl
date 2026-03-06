#=
MILP-Based Virtual Population Calibration Example

This script demonstrates Step 5 of the ISCT workflow:
"Select Virtual Patients and Calibrate Vpop"

The MILP algorithm selects a subset of virtual patients whose distribution
matches a target clinical distribution (e.g., from published clinical trials).

Two calibration modes are demonstrated:
1. Calibration to empirical data (using histogram binning)
2. Calibration to pre-specified distribution (using published percentages)

Reference:
    Section 2, Step 5 of the ISCT workflow paper (Figure 3)
=#

using .ISCTWorkflow
using DataFrames
using DataFramesMeta
using Statistics
using Random
using AlgebraOfGraphics
using CairoMakie

#=============================================================================
# Step 1: Generate or Load Virtual Population
=============================================================================#

println("="^60)
println("Step 1: Generating Virtual Population")
println("="^60)

# For this demo, we'll generate a synthetic VP with known distribution
# In practice, you would load from CSV or generate using copula sampling

Random.seed!(42)

# Create a VP with HBsAg values following a log-normal-like distribution
# This simulates the uncalibrated model predictions
n_vps = 50000
vpop = DataFrame(
    id = 1:n_vps,
    # HBsAg values in log10(IU/mL) - broad distribution before calibration
    HBsAg = randn(n_vps) .* 2.5 .+ 3.0  # Mean ~3.0 (1000 IU/mL), wide spread
)

println("Generated $(nrow(vpop)) virtual patients")
println("HBsAg range: $(round(minimum(vpop.HBsAg), digits = 2)) to $(round(maximum(vpop.HBsAg), digits = 2)) log10(IU/mL)")
println("HBsAg mean: $(round(mean(vpop.HBsAg), digits = 2)) log10(IU/mL)")

#=============================================================================
# Step 2: Define Target Distribution
=============================================================================#

println("\n" * "="^60)
println("Step 2: Defining Target Distributions")
println("="^60)

# Method A: Target from clinical data (MBMA approach)
# Simulating clinical data from a normal distribution centered at log10(1250) IU/mL
clinical_data = randn(10000) .* 0.1 .+ log10(1250)
target_mbma = create_target_from_data(clinical_data; nbins = 20, variable_name = :HBsAg)

println("Method A - MBMA Target (from clinical data):")
println("  Mean: $(round(log10(1250), digits = 2)) log10(IU/mL) = 1250 IU/mL")
println("  Number of bins: $(length(target_mbma.percentages))")

# Method B: Target from published trial (Everest approach)
# Pre-specified bin edges and percentages from Everest trial
everest_edges = [
    log10(0.03),   # Below LOQ
    log10(0.05),   # LOQ
    log10(100),    # 100 IU/mL
    log10(200),    # 200 IU/mL
    log10(500),    # 500 IU/mL
    log10(1000),   # 1000 IU/mL
    log10(1500),   # 1500 IU/mL
    log10(10000),   # 10000 IU/mL
]
everest_percentages = [0.0, 37.5, 10.9, 19.1, 21.5, 11.0, 0.0]  # Sum = 100

target_everest = create_target_from_specification(
    everest_edges, everest_percentages; variable_name = :HBsAg
)

println("\nMethod B - Everest Target (pre-specified):")
println("  Bins:")
for (i, pct) in enumerate(target_everest.percentages)
    lower = round(10^target_everest.edges[i], digits = 1)
    upper = round(10^target_everest.edges[i + 1], digits = 1)
    println("    $lower - $upper IU/mL: $(pct)%")
end

#=============================================================================
# Step 3: Run MILP Calibration
=============================================================================#

println("\n" * "="^60)
println("Step 3: Running MILP Calibration")
println("="^60)

# Calibration to MBMA target
println("\n--- Calibrating to MBMA target ---")
result_mbma = solve_milp_calibration(
    vpop.HBsAg,
    target_mbma,
    0.1;  # 10% tolerance per bin
    time_limit = 30.0
)
print_calibration_summary(result_mbma)

# Calibration to Everest target
println("\n--- Calibrating to Everest target ---")
result_everest = solve_milp_calibration(
    vpop.HBsAg,
    target_everest,
    0.2;  # 20% tolerance (more flexible for categorical bins)
    time_limit = 30.0
)
print_calibration_summary(result_everest)

#=============================================================================
# Step 4: Pareto Front Analysis
=============================================================================#

println("\n" * "="^60)
println("Step 4: Finding Pareto Front (nbins, epsilon)")
println("="^60)

# Find Pareto-optimal (nbins, epsilon) combinations
println("Searching for Pareto front...")
pareto_front = find_pareto_front(
    vpop.HBsAg,
    clinical_data;
    nbins_range = 10:5:40,
    epsilon_range = (0.05, 0.3),
    n_epsilon_samples = 8,
    time_limit_per_solve = 5.0
)

println("Found $(length(pareto_front)) Pareto-optimal points")

if !isempty(pareto_front)
    println("\nPareto Front:")
    println("-"^50)
    for (i, p) in enumerate(pareto_front[1:min(10, length(pareto_front))])
        println(
            "  $i. nbins=$(p.nbins), ε=$(round(p.epsilon, digits = 3)): " *
                "$(p.n_selected) VPs, $(round(p.mean_error, digits = 2))% error"
        )
    end

    # Select optimal point using knee method
    optimal = select_optimal_pareto_point(pareto_front; method = :knee)
    println("\nOptimal point (knee method):")
    println("  nbins = $(optimal.nbins)")
    println("  epsilon = $(round(optimal.epsilon, digits = 3))")
    println("  Selected VPs = $(optimal.n_selected)")
    println("  Mean error = $(round(optimal.mean_error, digits = 2))%")
end

#=============================================================================
# Step 5: Create Calibrated Virtual Population
=============================================================================#

println("\n" * "="^60)
println("Step 5: Creating Calibrated Virtual Population")
println("="^60)

# Use the MBMA calibration result
vpop_calibrated_mbma = vpop[result_mbma.selected_indices, :]
vpop_calibrated_mbma.id = 1:nrow(vpop_calibrated_mbma)

# Use the Everest calibration result
vpop_calibrated_everest = vpop[result_everest.selected_indices, :]
vpop_calibrated_everest.id = 1:nrow(vpop_calibrated_everest)

println("Calibrated VP (MBMA): $(nrow(vpop_calibrated_mbma)) patients")
println("  HBsAg mean: $(round(mean(vpop_calibrated_mbma.HBsAg), digits = 2)) log10(IU/mL)")
println("  HBsAg range: $(round(minimum(vpop_calibrated_mbma.HBsAg), digits = 2)) to $(round(maximum(vpop_calibrated_mbma.HBsAg), digits = 2))")

println("\nCalibrated VP (Everest): $(nrow(vpop_calibrated_everest)) patients")
println("  HBsAg mean: $(round(mean(vpop_calibrated_everest.HBsAg), digits = 2)) log10(IU/mL)")
println("  HBsAg range: $(round(minimum(vpop_calibrated_everest.HBsAg), digits = 2)) to $(round(maximum(vpop_calibrated_everest.HBsAg), digits = 2))")

#=============================================================================
# Step 6: Visualization
=============================================================================#

println("\n" * "="^60)
println("Step 6: Creating Visualizations")
println("="^60)

# Figure 1: Before vs After Calibration (MBMA)
fig1 = Figure(size = (800, 600))

ax1a = Axis(
    fig1[1, 1],
    xlabel = "log₁₀(HBsAg) IU/mL",
    ylabel = "Percentage (%)",
    title = "Before Calibration (Original VP)"
)
hist!(ax1a, vpop.HBsAg, bins = 30, normalization = :probability, color = (:blue, 0.5))
xlims!(ax1a, -2, 10)

ax1b = Axis(
    fig1[2, 1],
    xlabel = "log₁₀(HBsAg) IU/mL",
    ylabel = "Percentage (%)",
    title = "After Calibration (MBMA Target)"
)
hist!(
    ax1b, vpop_calibrated_mbma.HBsAg, bins = length(target_mbma.percentages),
    normalization = :probability, color = (:green, 0.5)
)
vlines!(ax1b, [log10(1250)], color = :red, linestyle = :dash, label = "Target mean")
xlims!(ax1b, 2.5, 4.0)

ax1c = Axis(
    fig1[3, 1],
    xlabel = "log₁₀(HBsAg) IU/mL",
    ylabel = "Percentage (%)",
    title = "Target Distribution (Clinical Data)"
)
hist!(ax1c, clinical_data, bins = 30, normalization = :probability, color = (:orange, 0.5))
xlims!(ax1c, 2.5, 4.0)

Label(fig1[0, :], "MILP Calibration: MBMA Target", fontsize = 18)

save(joinpath(@__DIR__, "..", "outputs", "calibration_mbma.png"), fig1)
println("Saved: outputs/calibration_mbma.png")

# Figure 2: Everest Calibration (Stacked Bar Comparison)
fig2 = Figure(size = (700, 500))

# Calculate percentages for each population
n_bins = length(target_everest.percentages)
bin_labels = ["0.05-100", "100-200", "200-500", "500-1000", "1000-1500"]

# Count VPs in each bin
function count_bins(values, edges)
    counts = zeros(length(edges) - 1)
    for v in values
        for b in 1:length(counts)
            if v > edges[b] && v <= edges[b + 1]
                counts[b] += 1
                break
            end
        end
    end
    return 100 .* counts ./ length(values)
end

pct_original = count_bins(vpop.HBsAg, everest_edges)
pct_calibrated = count_bins(vpop_calibrated_everest.HBsAg, everest_edges)
pct_target = everest_percentages

ax2 = Axis(
    fig2[1, 1],
    xlabel = "HBsAg Range (IU/mL)",
    ylabel = "Percentage (%)",
    title = "Everest Trial Distribution Matching",
    xticks = (1:5, bin_labels),
    xticklabelrotation = π / 6
)

# Only plot bins 2-6 (skip the first and last which are 0%)
x_pos = 1:5
barwidth = 0.25

barplot!(ax2, x_pos .- barwidth, pct_original[2:6], width = barwidth, color = :blue, label = "Original VP")
barplot!(ax2, x_pos, pct_calibrated[2:6], width = barwidth, color = :green, label = "Calibrated VP")
barplot!(ax2, x_pos .+ barwidth, pct_target[2:6], width = barwidth, color = :orange, label = "Everest Target")

axislegend(ax2, position = :rt)

save(joinpath(@__DIR__, "..", "outputs", "calibration_everest.png"), fig2)
println("Saved: outputs/calibration_everest.png")

# Figure 3: Pareto Front
if !isempty(pareto_front)
    fig3 = Figure(size = (600, 500))
    ax3 = Axis(
        fig3[1, 1],
        xlabel = "Number of Selected VPs",
        ylabel = "Mean Distribution Error (%)",
        title = "Pareto Front: VPs vs Distribution Matching"
    )

    vps = [p.n_selected for p in pareto_front]
    errors = [p.mean_error for p in pareto_front]

    scatter!(ax3, vps, errors, color = :blue, markersize = 10, label = "Pareto points")

    # Highlight optimal point
    optimal = select_optimal_pareto_point(pareto_front; method = :knee)
    scatter!(
        ax3, [optimal.n_selected], [optimal.mean_error],
        color = :red, markersize = 15, marker = :star5, label = "Optimal (knee)"
    )

    axislegend(ax3, position = :rt)

    save(joinpath(@__DIR__, "..", "outputs", "calibration_pareto.png"), fig3)
    println("Saved: outputs/calibration_pareto.png")
end

#=============================================================================
# Step 7: Summary and Usage
=============================================================================#

println("\n" * "="^60)
println("MILP Calibration Complete!")
println("="^60)

println("\nKey Results:")
println("  - Original VP: $(nrow(vpop)) patients")
println("  - MBMA Calibrated: $(nrow(vpop_calibrated_mbma)) patients ($(round(100 * result_mbma.selection_rate, digits = 1))%)")
println("  - Everest Calibrated: $(nrow(vpop_calibrated_everest)) patients ($(round(100 * result_everest.selection_rate, digits = 1))%)")

println("\nUsage Notes:")
println("  1. The calibrated VPop can be used directly for VCT simulation")
println("  2. Increase epsilon for more VPs with less strict matching")
println("  3. Decrease epsilon for stricter matching with fewer VPs")
println("  4. Use Pareto front to find optimal epsilon/nbins trade-off")

println("\nNext Steps:")
println("  - Save calibrated populations: CSV.write(\"vpop_calibrated.csv\", vpop_calibrated)")
println("  - Run VCT with calibrated population")
println("  - Compare outcomes with uncalibrated population")
