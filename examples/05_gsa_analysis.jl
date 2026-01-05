#=
Global Sensitivity Analysis Example

This script demonstrates Step 7 of the ISCT workflow:
"Analyze VCT results with Global Sensitivity Analysis"

Two GSA analyses are demonstrated:
1. Tumor Burden Model - Sobol/eFAST analysis on response endpoints
2. HBV Model - eFAST analysis on treatment outcomes

The GSA framework supports:
- Sobol method (variance-based, first/total order indices)
- eFAST method (computationally efficient alternative)
- Parameter importance ranking
- Interaction effect quantification

Reference:
    Section 2, Step 7 of the ISCT workflow paper
=#

using .ISCTWorkflow
using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

#=============================================================================
# Part 1: Tumor Burden GSA
=============================================================================#

println("=" ^ 70)
println("Part 1: Tumor Burden Global Sensitivity Analysis")
println("=" ^ 70)

# Define parameter ranges based on Qi & Cao (2023)
println("\nParameter Ranges for GSA:")
for param in TUMOR_BURDEN_GSA_PARAMS
    println("  $(param.name): [$(param.lower), $(param.upper)]")
end

# Run eFAST analysis (faster than Sobol)
println("\nRunning eFAST analysis on Tumor Burden model...")
println("  Outputs: final_tumor, auc_tumor")
println("  Samples: 500")

tb_gsa = run_tumor_burden_gsa(
    method = :efast,
    n_samples = 500,
    observation_times = 0:7:126,  # 18 weeks
    treatment = 1,                 # Treatment arm
    outputs = [:final_tumor, :auc_tumor]
)

# Print results
print_gsa_summary(tb_gsa)

# Get ranked summary
summary_tb = summarize_gsa(tb_gsa)
println("\nDetailed Rankings:")
display(summary_tb)

#=============================================================================
# Part 2: HBV Model GSA
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 2: HBV Model Global Sensitivity Analysis")
println("=" ^ 70)

# Define parameter ranges for HBV
println("\nParameter Ranges for HBV GSA:")
for param in HBV_GSA_PARAMS
    println("  $(param.name): [$(param.lower), $(param.upper)]")
end

# Run eFAST analysis
println("\nRunning eFAST analysis on HBV model...")
println("  Treatment: NUC + IFN combination")
println("  Outputs: final_hbsag, final_viral, hbsag_nadir")
println("  Samples: 300 (reduced for demo)")

hbv_gsa = run_hbv_gsa(
    method = :efast,
    n_samples = 300,
    treatment = :NUC_IFN,
    observation_times = 0:7:336,  # 48 weeks
    outputs = [:final_hbsag, :final_viral, :hbsag_nadir]
)

# Print results
print_gsa_summary(hbv_gsa)

# Get ranked summary
summary_hbv = summarize_gsa(hbv_gsa)
println("\nDetailed Rankings:")
display(summary_hbv)

#=============================================================================
# Part 3: Treatment Comparison (NUC vs IFN)
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 3: Treatment-Specific GSA Comparison")
println("=" ^ 70)

# Compare parameter importance across treatments
println("\nRunning GSA for NUC-only treatment...")
gsa_nuc = run_hbv_gsa(
    method = :efast,
    n_samples = 200,
    treatment = :NUC,
    outputs = [:final_hbsag, :final_viral]
)

println("\nRunning GSA for IFN-only treatment...")
gsa_ifn = run_hbv_gsa(
    method = :efast,
    n_samples = 200,
    treatment = :IFN,
    outputs = [:final_hbsag, :final_viral]
)

# Compare influential parameters
influential_nuc = get_influential_params(gsa_nuc; threshold=0.1)
influential_ifn = get_influential_params(gsa_ifn; threshold=0.1)

println("\nInfluential Parameters (Total Order ≥ 0.1):")
println("  NUC treatment: $(join(influential_nuc, \", \"))")
println("  IFN treatment: $(join(influential_ifn, \", \"))")

#=============================================================================
# Part 4: Visualization
=============================================================================#

println("\n" * "=" ^ 70)
println("Part 4: Creating Visualizations")
println("=" ^ 70)

# Figure 1: Tumor Burden GSA - Bar Chart
fig1 = Figure(size = (800, 500))

# Get data for final_tumor
tb_final = @chain summary_tb begin
    @subset(:output .== :final_tumor)
    @orderby(-:total_order)
end

ax1 = Axis(fig1[1, 1],
    xlabel = "Parameter",
    ylabel = "Sensitivity Index",
    title = "Tumor Burden GSA: Final Tumor Size",
    xticks = (1:nrow(tb_final), string.(tb_final.parameter))
)

x_pos = 1:nrow(tb_final)
barwidth = 0.35

barplot!(ax1, x_pos .- barwidth/2, tb_final.first_order,
         width=barwidth, color=:blue, label="First Order")
barplot!(ax1, x_pos .+ barwidth/2, tb_final.total_order,
         width=barwidth, color=:orange, label="Total Order")

axislegend(ax1, position=:rt)
hlines!(ax1, [0.1], color=:red, linestyle=:dash, label="Threshold")

save(joinpath(@__DIR__, "..", "outputs", "gsa_tumor_burden.png"), fig1)
println("Saved: outputs/gsa_tumor_burden.png")

# Figure 2: HBV GSA - Grouped Bar Chart
fig2 = Figure(size = (900, 600))

# Get data for final_hbsag
hbv_hbsag = @chain summary_hbv begin
    @subset(:output .== :final_hbsag)
    @orderby(-:total_order)
end

ax2 = Axis(fig2[1, 1],
    xlabel = "Parameter",
    ylabel = "Sensitivity Index",
    title = "HBV GSA: Final HBsAg Level",
    xticks = (1:nrow(hbv_hbsag), string.(hbv_hbsag.parameter)),
    xticklabelrotation = π/4
)

x_pos2 = 1:nrow(hbv_hbsag)
barwidth2 = 0.35

barplot!(ax2, x_pos2 .- barwidth2/2, hbv_hbsag.first_order,
         width=barwidth2, color=:blue, label="First Order")
barplot!(ax2, x_pos2 .+ barwidth2/2, hbv_hbsag.total_order,
         width=barwidth2, color=:orange, label="Total Order")

axislegend(ax2, position=:rt)
hlines!(ax2, [0.1], color=:red, linestyle=:dash)

save(joinpath(@__DIR__, "..", "outputs", "gsa_hbv_hbsag.png"), fig2)
println("Saved: outputs/gsa_hbv_hbsag.png")

# Figure 3: Treatment Comparison Heatmap
fig3 = Figure(size = (700, 500))

# Create comparison matrix
params = collect(HBV_GSA_PARAMS)
param_names = [string(p.name) for p in params]
n_params = length(params)

# Extract total order indices
function get_total_order_vector(gsa_result, output_name)
    summary = summarize_gsa(gsa_result)
    output_data = filter(row -> row.output == output_name, summary)

    indices = zeros(n_params)
    for (i, p) in enumerate(params)
        row = filter(r -> r.parameter == p.name, output_data)
        if nrow(row) > 0
            indices[i] = row[1, :total_order]
        end
    end
    return indices
end

nuc_indices = get_total_order_vector(gsa_nuc, :final_hbsag)
ifn_indices = get_total_order_vector(gsa_ifn, :final_hbsag)
combo_indices = get_total_order_vector(hbv_gsa, :final_hbsag)

# Create grouped bar chart
ax3 = Axis(fig3[1, 1],
    xlabel = "Parameter",
    ylabel = "Total Order Index",
    title = "HBV GSA: Treatment Comparison (Final HBsAg)",
    xticks = (1:n_params, param_names),
    xticklabelrotation = π/4
)

barwidth3 = 0.25
offsets = [-barwidth3, 0, barwidth3]

barplot!(ax3, (1:n_params) .+ offsets[1], nuc_indices,
         width=barwidth3, color=:blue, label="NUC")
barplot!(ax3, (1:n_params) .+ offsets[2], ifn_indices,
         width=barwidth3, color=:orange, label="IFN")
barplot!(ax3, (1:n_params) .+ offsets[3], combo_indices,
         width=barwidth3, color=:green, label="NUC+IFN")

axislegend(ax3, position=:rt)

save(joinpath(@__DIR__, "..", "outputs", "gsa_treatment_comparison.png"), fig3)
println("Saved: outputs/gsa_treatment_comparison.png")

# Figure 4: Interaction Effects
fig4 = Figure(size = (700, 500))

# Plot interaction (total - first order) for HBV model
hbv_interactions = @chain summary_hbv begin
    @subset(:output .== :final_hbsag)
    @orderby(-:interaction)
end

ax4 = Axis(fig4[1, 1],
    xlabel = "Parameter",
    ylabel = "Interaction Effect (Total - First Order)",
    title = "HBV GSA: Parameter Interactions",
    xticks = (1:nrow(hbv_interactions), string.(hbv_interactions.parameter)),
    xticklabelrotation = π/4
)

barplot!(ax4, 1:nrow(hbv_interactions), hbv_interactions.interaction,
         color=:purple)

save(joinpath(@__DIR__, "..", "outputs", "gsa_interactions.png"), fig4)
println("Saved: outputs/gsa_interactions.png")

#=============================================================================
# Summary
=============================================================================#

println("\n" * "=" ^ 70)
println("GSA Analysis Complete!")
println("=" ^ 70)

println("\nKey Findings:")

println("\nTumor Burden Model:")
top_tb = first(tb_final, 3)
for row in eachrow(top_tb)
    println("  - $(row.parameter): Total Order = $(round(row.total_order, digits=3))")
end

println("\nHBV Model (NUC+IFN):")
top_hbv = first(hbv_hbsag, 3)
for row in eachrow(top_hbv)
    println("  - $(row.parameter): Total Order = $(round(row.total_order, digits=3))")
end

println("\nInterpretation Guide:")
println("  - First Order Index: Direct effect of parameter variation")
println("  - Total Order Index: Combined direct + interaction effects")
println("  - Interaction Effect: Contribution from parameter interactions")
println("  - Threshold (0.1): Parameters with total order > 0.1 are influential")

println("\nUsage Recommendations:")
println("  1. Focus calibration efforts on high-sensitivity parameters")
println("  2. Parameters with high interaction effects may need joint calibration")
println("  3. Low-sensitivity parameters can use wider ranges or be fixed")

println("\nOutput files saved to outputs/")
