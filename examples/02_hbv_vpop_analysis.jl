#=
HBV Virtual Population Analysis Example

This script demonstrates the HBV virtual population workflow:
1. Loading pre-sampled virtual patients from CSV
2. Computing parameter statistics and correlations
3. Generating new VPs using Gaussian copulas
4. Summarizing and comparing populations

Note: Full VCT simulation with the 11-ODE model is computationally intensive.
This example focuses on the parameter sampling and analysis workflow.

Reference:
    Section 3.2 of the ISCT workflow paper - HBV QSP Model
=#

using .ISCTWorkflow
using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

#=============================================================================
# Step 1: Load Pre-Sampled Virtual Population
=============================================================================#

println("=" ^ 60)
println("Step 1: Loading Pre-Sampled Virtual Population")
println("=" ^ 60)

# Path to parameter CSV (adjust as needed)
param_csv = joinpath(@__DIR__, "..", "..", "ISCT SoC and MBMA comparison", "Parameters_Naive.csv")

if isfile(param_csv)
    println("Loading from: $param_csv")
    vpop_full = load_hbv_vpop(param_csv)
    println("Loaded $(nrow(vpop_full)) virtual patients")

    # Display first few rows
    println("\nFirst 5 virtual patients (estimated parameters):")
    display(first(extract_estimated_params(vpop_full), 5))
else
    println("Parameter CSV not found at: $param_csv")
    println("Creating synthetic demo population instead...")

    # Create a demo population with known statistics
    # These are approximate values based on the expected distributions
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

    # Identity correlation for demo
    demo_corr = Matrix{Float64}(I, 9, 9)

    vpop_full = generate_hbv_vpop(10000, demo_specs, demo_corr; seed=42)
    println("Generated $(nrow(vpop_full)) synthetic virtual patients")
end

#=============================================================================
# Step 2: Compute Parameter Statistics
=============================================================================#

println("\n" * "=" ^ 60)
println("Step 2: Computing Parameter Statistics")
println("=" ^ 60)

# Compute statistics
stats = compute_hbv_stats(vpop_full)
print_hbv_stats(stats)

# Create summary table
summary_stats = summarize_hbv_vpop(vpop_full)
println("\nDetailed Summary Statistics:")
display(summary_stats)

#=============================================================================
# Step 3: Generate New Virtual Population Using Copulas
=============================================================================#

println("\n" * "=" ^ 60)
println("Step 3: Generating New VP Using Copulas")
println("=" ^ 60)

# Create parameter specs from computed statistics
param_specs = create_hbv_param_specs(stats)

# Generate new population preserving correlations
n_new = 5000
vpop_new = generate_hbv_vpop(
    n_new,
    param_specs,
    stats.correlation;
    seed = 123,
    include_fixed = true
)

println("Generated $(nrow(vpop_new)) new virtual patients")

# Compare statistics
println("\nOriginal vs New Population Means:")
for name in HBV_ESTIMATED_PARAMS
    orig_mean = mean(vpop_full[!, name])
    new_mean = mean(vpop_new[!, name])
    diff_pct = 100 * abs(new_mean - orig_mean) / abs(orig_mean)
    println("  $(name): Original=$(round(orig_mean, digits=3)), New=$(round(new_mean, digits=3)), Diff=$(round(diff_pct, digits=1))%")
end

# Verify correlations preserved
new_stats = compute_hbv_stats(vpop_new)
println("\nCorrelation Preservation Check:")
max_corr_diff = maximum(abs.(stats.correlation - new_stats.correlation))
println("  Max correlation difference: $(round(max_corr_diff, digits=4))")
println("  Correlations preserved: $(max_corr_diff < 0.05 ? "YES" : "NO")")

#=============================================================================
# Step 4: Subsampling for VCT
=============================================================================#

println("\n" * "=" ^ 60)
println("Step 4: Subsampling for Faster VCT")
println("=" ^ 60)

# Subsample for computational efficiency
vpop_subsample = subsample_hbv_vpop(vpop_full, 1000; seed=42)
println("Subsampled $(nrow(vpop_subsample)) patients for VCT")

# Show subsample statistics
subsample_summary = summarize_hbv_vpop(vpop_subsample)
println("\nSubsample Summary (first 5 parameters):")
display(first(subsample_summary, 5))

#=============================================================================
# Step 5: Visualization
=============================================================================#

println("\n" * "=" ^ 60)
println("Step 5: Creating Visualizations")
println("=" ^ 60)

# Plot 1: Parameter Distributions
fig1 = Figure(size = (1000, 600))

# Select key parameters to plot
plot_params = [:beta, :p_S, :m, :epsNUC, :epsIFN, :k_D]

for (i, param) in enumerate(plot_params)
    row = div(i - 1, 3) + 1
    col = mod(i - 1, 3) + 1

    ax = Axis(fig1[row, col], xlabel = string(param), ylabel = "Density")
    hist!(ax, vpop_full[!, param], bins = 50, color = (:blue, 0.5), normalization = :pdf)

    # Add mean line
    vlines!(ax, [mean(vpop_full[!, param])], color = :red, linestyle = :dash)
end

Label(fig1[0, :], "HBV Estimated Parameter Distributions", fontsize = 16)
fig1
save(joinpath(@__DIR__, "..", "outputs", "hbv_param_distributions.png"), fig1)
println("Saved: outputs/hbv_param_distributions.png")

# Plot 2: Parameter Correlations Heatmap
fig2 = Figure(size = (600, 500))
ax2 = Axis(
    fig2[1, 1],
    title = "HBV Parameter Correlation Matrix",
    xlabel = "Parameter",
    ylabel = "Parameter",
    xticks = (1:9, string.(HBV_ESTIMATED_PARAMS)),
    yticks = (1:9, string.(HBV_ESTIMATED_PARAMS)),
    xticklabelrotation = π/4
)

hm = heatmap!(ax2, stats.correlation, colormap = :RdBu, colorrange = (-1, 1))
Colorbar(fig2[1, 2], hm, label = "Correlation")
fig2
save(joinpath(@__DIR__, "..", "outputs", "hbv_correlation_matrix.png"), fig2)
println("Saved: outputs/hbv_correlation_matrix.png")

# Plot 3: Key Parameter Scatter (beta vs p_S)
fig3 = Figure(size = (500, 500))
ax3 = Axis(
    fig3[1, 1],
    xlabel = "beta (Infection rate, log10)",
    ylabel = "p_S (HBsAg production, log10)",
    title = "Parameter Correlation: beta vs p_S"
)

# Subsample for visualization
n_plot = min(2000, nrow(vpop_full))
scatter!(
    ax3,
    vpop_full.beta[1:n_plot],
    vpop_full.p_S[1:n_plot],
    color = (:blue, 0.2),
    markersize = 4
)

# Add correlation annotation
r = round(cor(vpop_full.beta, vpop_full.p_S), digits=2)
text!(ax3, "r = $r", position = (minimum(vpop_full.beta[1:n_plot]) + 1, maximum(vpop_full.p_S[1:n_plot]) - 0.2))
fig3
save(joinpath(@__DIR__, "..", "outputs", "hbv_beta_ps_scatter.png"), fig3)
println("Saved: outputs/hbv_beta_ps_scatter.png")

# Plot 4: Treatment Efficacy Parameter Distributions
fig4 = Figure(size = (800, 300))

ax4a = Axis(fig4[1, 1], xlabel = "epsNUC (log10)", ylabel = "Density", title = "NUC Efficacy")
hist!(ax4a, vpop_full.epsNUC, bins = 50, color = (:orange, 0.6), normalization = :pdf)

ax4b = Axis(fig4[1, 2], xlabel = "epsIFN (log10)", ylabel = "Density", title = "IFN Efficacy")
hist!(ax4b, vpop_full.epsIFN, bins = 50, color = (:green, 0.6), normalization = :pdf)

ax4c = Axis(fig4[1, 3], xlabel = "r_E_IFN", ylabel = "Density", title = "IFN Effect on E cells")
hist!(ax4c, vpop_full.r_E_IFN, bins = 50, color = (:purple, 0.6), normalization = :pdf)
fig4
save(joinpath(@__DIR__, "..", "outputs", "hbv_treatment_params.png"), fig4)
println("Saved: outputs/hbv_treatment_params.png")

#=============================================================================
# Step 6: Demonstrate Model and Dosing Schedule
=============================================================================#

println("\n" * "=" ^ 60)
println("Step 6: HBV Model and Dosing Schedule Demo")
println("=" ^ 60)

# Show available models
println("Available HBV models:")
println("  - hbv_model: Full model with random effects")
println("  - For VCT simulation: use zero_randeffs() or pass individual random effects to simobs()")

# Show treatment phases
println("\nTreatment Phases:")
println("  Untreated: $(HBV_PHASES.untreated) days ($(HBV_PHASES.untreated ÷ 365) years)")
println("  NUC background: $(HBV_PHASES.nuc_background) days ($(HBV_PHASES.nuc_background ÷ 365) years)")
println("  Treatment: $(HBV_PHASES.treatment) days ($(HBV_PHASES.treatment ÷ 7) weeks)")
println("  Off-treatment: $(HBV_PHASES.off_treatment) days ($(HBV_PHASES.off_treatment ÷ 7) weeks)")

# Create dosing schedules for different arms
println("\nTreatment Arms:")
println("  0: Control (no treatment)")
println("  1: NUC only")
println("  2: IFN only")
println("  3: NUC + IFN combination")

# Show example dosing schedule
schedule_combo = create_hbv_dosing_schedule(HBV_TREATMENT.combo; suppressed=true)
println("\nExample: Combo treatment (suppressed patient)")
println("  Total time points: $(length(schedule_combo.times))")
println("  First treatment time: $(findfirst(x -> x > 0, schedule_combo.dNUC)) days")

# Show clinical endpoints
println("\nClinical Endpoints:")
println("  HBsAg LOQ: $(LOQ_HBsAg) IU/mL (log10 = $(round(log10(LOQ_HBsAg), digits=2)))")
println("  Viral LOQ: $(LOQ_V) copies/mL (log10 = $(round(LOG_LOQ_V, digits=2)))")
println("  Functional Cure: HBsAg < LOQ AND Viral Load < LOQ for ≥24 weeks")

println("\n" * "=" ^ 60)
println("HBV Virtual Population Analysis Complete!")
println("=" ^ 60)
println("\nKey Outputs:")
println("  - Parameter statistics computed from Vpop")
println("  - New population generated using Gaussian copulas")
println("  - Correlations preserved between parameters")
println("  - Visualizations saved to outputs/")

println("\nNote: Full HBV VCT simulation with 11 ODEs is computationally intensive.")
println("For large-scale trials, consider using parallel simulation or subsampling.")
