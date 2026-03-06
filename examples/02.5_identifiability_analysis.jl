#=
Example 02.5: Structural Identifiability Analysis

This example demonstrates how to assess structural identifiability of
the tumor burden and HBV models using StructuralIdentifiability.jl.

Structural identifiability is a prerequisite check BEFORE attempting parameter
estimation. It determines whether model parameters can be uniquely estimated
from input-output data, assuming perfect (noise-free) data.

This corresponds to Step 3 of the ISCT workflow paper:
    "Conduct Sensitivity and Identifiability Analysis"

Sections:
1. Tumor Burden - Single scenario analysis
2. Tumor Burden - Multi-scenario comparison
3. Tumor Burden - Identifiable functions & reparameterization
4. HBV - Local identifiability (faster)
5. HBV - Progressive measurement scenarios
6. HBV - Multi-experiment identifiability
7. Visualization & Summary
=#

using ISCTWorkflow
using DataFrames

# Create output directory
mkpath("outputs")

#=============================================================================
# 1. Tumor Burden Model: Single Scenario Analysis
=============================================================================#

println("\n" * "="^70)
println("  SECTION 1: TUMOR BURDEN - SINGLE SCENARIO ANALYSIS")
println("="^70)

# Extract the ODE system from the Pumas model
ode_sys = extract_ode_system(tumor_burden_model)

# Create measurement scenarios
tb_scenarios = create_tumor_burden_scenarios(ode_sys)

println("\nAvailable measurement scenarios for Tumor Burden model:")
for (name, scenario) in tb_scenarios
    println("  - $(name): $(scenario.description)")
    println("    Clinically relevant: $(scenario.clinically_relevant)")
end

# Analyze the clinically relevant scenario (total tumor size only)
println("\n--- Analyzing 'tumor_size_only' scenario ---")
result_tumor_only = assess_scenario_identifiability(
    ode_sys,
    tb_scenarios[:tumor_size_only];
    verbose = true
)

println("\nKey findings:")
println("  - Globally identifiable: $(result_tumor_only.globally_identifiable)")
println("  - Locally identifiable: $(result_tumor_only.locally_identifiable)")
println("  - Non-identifiable: $(result_tumor_only.nonidentifiable)")
println("  - All globally identifiable: $(result_tumor_only.all_globally_identifiable)")

#=============================================================================
# 2. Tumor Burden Model: Multi-Scenario Comparison
=============================================================================#

println("\n" * "="^70)
println("  SECTION 2: TUMOR BURDEN - MULTI-SCENARIO COMPARISON")
println("="^70)

# Run comprehensive analysis on all scenarios
report_tb = analyze_tumor_burden_identifiability(
    scenarios = [:tumor_size_only, :both_populations],
    include_global = true,
    include_local = true,
    find_functions = true,
    compute_reparameterization = true,
    verbose = true
)

# Print the summary table
println("\n--- Scenario Comparison Summary ---")
println(report_tb.summary)

# Generate and print recommendations
recommendations = generate_recommendations(report_tb.global_results)
println("\n--- Practical Recommendations ---")
for rec in recommendations
    println("  - $rec")
end

#=============================================================================
# 3. Tumor Burden: Identifiable Functions & Reparameterization
=============================================================================#

println("\n" * "="^70)
println("  SECTION 3: IDENTIFIABLE FUNCTIONS & REPARAMETERIZATION")
println("="^70)

# Find what parameter combinations ARE identifiable
# (useful when individual parameters are not)
funcs_result = find_scenario_identifiable_functions(
    ode_sys,
    tb_scenarios[:tumor_size_only];
    with_states = false,
    simplify = :standard,
    verbose = true
)

println("\nIdentifiable functions found: $(length(funcs_result.identifiable_functions))")

# Compute reparameterization
reparam_result = compute_scenario_reparameterization(
    ode_sys,
    tb_scenarios[:tumor_size_only];
    verbose = true
)

println("\nReparameterization recommendations:")
for rec in reparam_result.recommendations
    println("  - $rec")
end

#=============================================================================
# 4. HBV Model: Local Identifiability Analysis (Faster)
=============================================================================#

println("\n" * "="^70)
println("  SECTION 4: HBV - LOCAL IDENTIFIABILITY ANALYSIS")
println("="^70)

println("\nNote: HBV model is complex (11 ODEs, 31 parameters)")
println("Using local identifiability analysis for speed...")

# Extract HBV ODE system
hbv_ode_sys = extract_ode_system(hbv_model)

# Create measurement scenarios
hbv_scenarios = create_hbv_scenarios(hbv_ode_sys)

println("\nAvailable measurement scenarios for HBV model:")
for (name, scenario) in hbv_scenarios
    println("  - $(name): $(scenario.description)")
end

# Quick local analysis on standard clinical monitoring scenario
println("\n--- Local Analysis: HBsAg + Viral Load ---")
local_result = assess_local_scenario_identifiability(
    hbv_ode_sys,
    hbv_scenarios[:hbsag_and_viral];
    type = :SE,
    verbose = true
)

println("\nLocal identifiability summary:")
println("  - Identifiable: $(local_result.identifiable)")
println("  - Non-identifiable: $(local_result.nonidentifiable)")

#=============================================================================
# 5. HBV: Progressive Measurement Scenarios
=============================================================================#

println("\n" * "="^70)
println("  SECTION 5: HBV - PROGRESSIVE MEASUREMENT SCENARIOS")
println("="^70)

println("\nComparing identifiability as we add more measurements...")

# Analyze progressively more complete measurement sets
local_results = Dict{Symbol, LocalIdentifiabilityResult}()

for scenario_name in [:hbsag_only, :viral_only, :hbsag_and_viral, :hbsag_viral_alt]
    println("\n--- Scenario: $(scenario_name) ---")
    local_results[scenario_name] = assess_local_scenario_identifiability(
        hbv_ode_sys,
        hbv_scenarios[scenario_name];
        type = :SE,
        verbose = false  # Quiet for comparison
    )

    result = local_results[scenario_name]
    n_id = length(result.identifiable)
    n_total = n_id + length(result.nonidentifiable)
    println("  Identifiable: $(n_id)/$(n_total)")
    if !isempty(result.nonidentifiable)
        println("  Non-identifiable: $(result.nonidentifiable)")
    end
end

# Compare results
comparison_df = compare_local_scenarios(local_results)
println("\n--- Progressive Scenario Comparison ---")
println(comparison_df)

#=============================================================================
# 6. HBV: Multi-Experiment Identifiability
=============================================================================#

println("\n" * "="^70)
println("  SECTION 6: HBV - MULTI-EXPERIMENT IDENTIFIABILITY")
println("="^70)

println("\nMulti-experiment (ME) analysis can improve identifiability")
println("by combining data from multiple experimental conditions...")

# Compare single-experiment vs multi-experiment
println("\n--- Single Experiment (SE) vs Multi-Experiment (ME) ---")

se_result = assess_local_scenario_identifiability(
    hbv_ode_sys,
    hbv_scenarios[:hbsag_and_viral];
    type = :SE,
    verbose = false
)

me_result = assess_local_scenario_identifiability(
    hbv_ode_sys,
    hbv_scenarios[:hbsag_and_viral];
    type = :ME,
    verbose = false
)

println("\nSingle Experiment (SE):")
println("  Identifiable: $(length(se_result.identifiable))")
println("  Non-identifiable: $(se_result.nonidentifiable)")

println("\nMulti-Experiment (ME):")
println("  Identifiable: $(length(me_result.identifiable))")
println("  Non-identifiable: $(me_result.nonidentifiable)")

#=============================================================================
# 7. Comprehensive HBV Analysis
=============================================================================#

println("\n" * "="^70)
println("  SECTION 7: COMPREHENSIVE HBV ANALYSIS")
println("="^70)

# Run full analysis (local only for speed)
report_hbv = analyze_hbv_identifiability(
    scenarios = [:hbsag_and_viral],
    include_global = false,  # Skip global (slow)
    include_local = true,
    find_functions = false,  # Skip for speed
    compute_reparameterization = false,
    verbose = true
)

# Print comprehensive report
print_identifiability_report(report_hbv)

#=============================================================================
# 8. Visualization
=============================================================================#

println("\n" * "="^70)
println("  SECTION 8: VISUALIZATION")
println("="^70)

# Create comparison heatmap for tumor burden scenarios
if length(report_tb.global_results) >= 2
    println("\nCreating identifiability comparison plot...")

    fig = plot_identifiability_comparison(report_tb.global_results)
    save_figure(fig, "outputs/identifiability_tumor_burden_comparison.png")
end

# Create summary plot
println("\nCreating identifiability summary plot...")
fig_summary = plot_identifiability_summary(report_tb)
save_figure(fig_summary, "outputs/identifiability_tumor_burden_summary.png")

#=============================================================================
# Summary and Practical Guidance
=============================================================================#

println("\n" * "="^70)
println("  SUMMARY: PRACTICAL GUIDANCE FOR ISCT WORKFLOW")
println("="^70)

println(
    """

    TUMOR BURDEN MODEL:
    - All 3 parameters (f, g, k) should be structurally identifiable
    - Clinical measurement (total tumor size) is sufficient
    - Can proceed with parameter estimation in Step 4

    HBV QSP MODEL:
    - Complex model requires careful measurement planning
    - Standard clinical monitoring (HBsAg + viral load) provides good identifiability
    - Some parameters may require:
      * Additional measurements (ALT, etc.)
      * Informative priors from literature
      * Fixed values from mechanistic studies

    NEXT STEPS:
    1. If all parameters are globally identifiable -> Proceed to Step 4 (sampling)
    2. If some parameters are non-identifiable:
       a. Consider reparameterization (identifiable combinations)
       b. Fix parameters to literature values
       c. Use informative priors in Bayesian estimation
       d. Add additional measurements if feasible

    For more details, see Section 2, Step 3 of the ISCT workflow paper.
    """
)

println("\n--- Example Complete ---")
