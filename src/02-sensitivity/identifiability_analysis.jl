#=
Structural Identifiability Analysis Module

Implements Step 3 of the ISCT workflow: "Conduct Sensitivity and Identifiability Analysis"

This module provides utilities for assessing structural identifiability of
the tumor burden and HBV models using StructuralIdentifiability.jl.

Structural identifiability determines whether model parameters can be uniquely
estimated from input-output data, assuming:
  - Perfect (noise-free) data
  - Infinite time horizon observations
  - Known model structure

This is a prerequisite check BEFORE attempting parameter estimation. Even with
perfect data, structurally non-identifiable parameters cannot be estimated.

Identifiability classifications:
  - Globally identifiable: Unique parameter value exists
  - Locally identifiable: Finite number of parameter values possible
  - Non-identifiable: Infinite parameter values yield same output

Reference:
    Section 2, Step 3 of the ISCT workflow paper
=#

using StructuralIdentifiability
using Pumas.ModelingToolkit: @variables, t_nounits
using DataFrames
# import Pkg; Pkg.add("OrderedCollections")
using OrderedCollections: OrderedDict

#=============================================================================
# Data Structures
=============================================================================#

"""
    MeasurementScenario

Defines what quantities can be measured in a clinical scenario.

# Fields
- `name::Symbol`: Unique identifier for the scenario
- `description::String`: Human-readable description
- `measured_quantities::Vector`: Vector of measurement equations
- `clinically_relevant::Bool`: Whether this scenario is clinically feasible
"""
struct MeasurementScenario
    name::Symbol
    description::String
    measured_quantities::Vector
    clinically_relevant::Bool

    function MeasurementScenario(
            name::Symbol,
            description::String,
            measured_quantities::Vector;
            clinically_relevant::Bool = true
        )
        return new(name, description, measured_quantities, clinically_relevant)
    end
end

"""
    IdentifiabilityResult

Container for global identifiability analysis results.

# Fields
- `scenario::MeasurementScenario`: The measurement scenario used
- `identifiability::OrderedDict`: Parameter => identifiability status mapping
- `globally_identifiable::Vector{Symbol}`: Parameters that are globally identifiable
- `locally_identifiable::Vector{Symbol}`: Parameters that are locally identifiable
- `nonidentifiable::Vector{Symbol}`: Parameters that are not identifiable
- `all_globally_identifiable::Bool`: Whether all parameters are globally identifiable
"""
struct IdentifiabilityResult
    scenario::MeasurementScenario
    identifiability::OrderedDict
    globally_identifiable::Vector{Symbol}
    locally_identifiable::Vector{Symbol}
    nonidentifiable::Vector{Symbol}
    all_globally_identifiable::Bool

    function IdentifiabilityResult(scenario::MeasurementScenario, identifiability::OrderedDict)
        globally_identifiable = Symbol[]
        locally_identifiable = Symbol[]
        nonidentifiable = Symbol[]

        for (param, status) in identifiability
            param_sym = Symbol(string(param))
            if status == :globally
                push!(globally_identifiable, param_sym)
            elseif status == :locally
                push!(locally_identifiable, param_sym)
            else
                push!(nonidentifiable, param_sym)
            end
        end

        all_global = all(v == :globally for v in values(identifiability))

        return new(
            scenario, identifiability, globally_identifiable, locally_identifiable,
            nonidentifiable, all_global
        )
    end
end

"""
    LocalIdentifiabilityResult

Container for local identifiability analysis results.

# Fields
- `scenario::MeasurementScenario`: The measurement scenario used
- `identifiability::OrderedDict`: Parameter => Bool mapping (identifiable or not)
- `identifiable::Vector{Symbol}`: Locally identifiable parameters
- `nonidentifiable::Vector{Symbol}`: Non-identifiable parameters
- `type::Symbol`: Type of analysis (:SE for single experiment, :ME for multi-experiment)
- `num_experiments::Int`: Number of experiments (for :ME type)
"""
struct LocalIdentifiabilityResult
    scenario::MeasurementScenario
    identifiability::OrderedDict
    identifiable::Vector{Symbol}
    nonidentifiable::Vector{Symbol}
    type::Symbol
    num_experiments::Int

    function LocalIdentifiabilityResult(
            scenario::MeasurementScenario,
            identifiability::OrderedDict;
            type::Symbol = :SE,
            num_experiments::Int = 1
        )
        identifiable = Symbol[]
        nonidentifiable = Symbol[]

        for (param, is_id) in identifiability
            param_sym = Symbol(string(param))
            if is_id
                push!(identifiable, param_sym)
            else
                push!(nonidentifiable, param_sym)
            end
        end

        return new(scenario, identifiability, identifiable, nonidentifiable, type, num_experiments)
    end
end

"""
    IdentifiableFunctionsResult

Container for identifiable functions analysis results.

# Fields
- `scenario::MeasurementScenario`: The measurement scenario used
- `identifiable_functions::Vector`: Identifiable parameter combinations
- `with_states::Bool`: Whether states were included in the analysis
- `simplified::Bool`: Whether simplification was applied
"""
struct IdentifiableFunctionsResult
    scenario::MeasurementScenario
    identifiable_functions::Vector
    with_states::Bool
    simplified::Bool
end

"""
    ReparameterizationResult

Container for reparameterization analysis results.

# Fields
- `scenario::MeasurementScenario`: The measurement scenario used
- `new_vars::Dict`: Mapping from new variables to original expressions
- `recommendations::Vector{String}`: Human-readable recommendations
"""
struct ReparameterizationResult
    scenario::MeasurementScenario
    new_vars::Dict
    recommendations::Vector{String}
end

"""
    ComprehensiveIdentifiabilityReport

Container for multi-scenario identifiability analysis.

# Fields
- `model_name::String`: Name of the model analyzed
- `global_results::Dict{Symbol, IdentifiabilityResult}`: Global results by scenario
- `local_results::Dict{Symbol, LocalIdentifiabilityResult}`: Local results by scenario
- `functions_results::Dict{Symbol, IdentifiableFunctionsResult}`: Identifiable functions by scenario
- `reparameterization_results::Dict{Symbol, ReparameterizationResult}`: Reparameterization by scenario
- `summary::DataFrame`: Summary comparison across scenarios
"""
struct ComprehensiveIdentifiabilityReport
    model_name::String
    global_results::Dict{Symbol, IdentifiabilityResult}
    local_results::Dict{Symbol, LocalIdentifiabilityResult}
    functions_results::Dict{Symbol, IdentifiableFunctionsResult}
    reparameterization_results::Dict{Symbol, ReparameterizationResult}
    summary::DataFrame
end

#=============================================================================
# Default Measurement Scenarios
=============================================================================#

# Create symbolic variables for output functions (must be functions of time)
@variables Nt(t_nounits) N_sens_out(t_nounits) N_res_out(t_nounits)
@variables log_HBsAg(t_nounits) log_V(t_nounits) log_ALT(t_nounits)
@variables log_E(t_nounits) T_out(t_nounits) I_out(t_nounits)

"""
    TUMOR_BURDEN_SCENARIOS

Default measurement scenarios for tumor burden model.
The model has 3 parameters (f, g, k) and 2 state variables (N_sens, N_res).
"""
const TUMOR_BURDEN_SCENARIOS = Dict{Symbol, Function}()

"""
    create_tumor_burden_scenarios(ode_system)

Create measurement scenarios for the tumor burden model.
Must be called with the ODESystem extracted from the Pumas model.

# Arguments
- `ode_system`: ModelingToolkit ODESystem from tumor_burden_model.sys

# Returns
Dict of MeasurementScenario objects keyed by scenario name
"""
function create_tumor_burden_scenarios(ode_system)
    # Create symbolic variables for outputs
    @variables Nt(t_nounits) N_sens_out(t_nounits) N_res_out(t_nounits)

    return Dict(
        :tumor_size_only => MeasurementScenario(
            :tumor_size_only,
            "Total tumor size only (clinical standard - CT/MRI measurement)",
            [Nt ~ ode_system.N_sens + ode_system.N_res],
            clinically_relevant = true
        ),
        :both_populations => MeasurementScenario(
            :both_populations,
            "Both sensitive and resistant populations (theoretical - not clinically feasible)",
            [N_sens_out ~ ode_system.N_sens, N_res_out ~ ode_system.N_res],
            clinically_relevant = false
        )
    )
end

"""
    HBV_ESTIMATED_PARAM_NAMES

Names of the 9 estimated parameters in the HBV model.
"""
const HBV_ESTIMATED_PARAM_NAMES = [
    :beta, :p_S, :m, :k_Z, :convE, :epsNUC, :epsIFN, :r_E_IFN, :k_D,
]

"""
    create_hbv_scenarios(ode_system)

Create measurement scenarios for the HBV model.
Must be called with the ODESystem extracted from the Pumas model.

# Arguments
- `ode_system`: ModelingToolkit ODESystem from hbv_model.sys

# Returns
Dict of MeasurementScenario objects keyed by scenario name
"""
function create_hbv_scenarios(ode_system)
    # Create symbolic variables for outputs
    @variables log_HBsAg(t_nounits) log_V(t_nounits) log_ALT(t_nounits)

    return Dict(
        :hbsag_only => MeasurementScenario(
            :hbsag_only,
            "HBsAg measurement only (surface antigen ELISA)",
            [log_HBsAg ~ log10(ode_system.S)],
            clinically_relevant = true
        ),
        :viral_only => MeasurementScenario(
            :viral_only,
            "Viral load (HBV DNA) only (PCR measurement)",
            [log_V ~ log10(ode_system.V)],
            clinically_relevant = true
        ),
        :hbsag_and_viral => MeasurementScenario(
            :hbsag_and_viral,
            "Both HBsAg and viral load (standard clinical monitoring)",
            [log_HBsAg ~ log10(ode_system.S), log_V ~ log10(ode_system.V)],
            clinically_relevant = true
        ),
        :hbsag_viral_alt => MeasurementScenario(
            :hbsag_viral_alt,
            "HBsAg, viral load, and ALT (comprehensive liver panel)",
            [
                log_HBsAg ~ log10(ode_system.S), log_V ~ log10(ode_system.V),
                log_ALT ~ log10(ode_system.Z),
            ],
            clinically_relevant = true
        ),
        :all_observables => MeasurementScenario(
            :all_observables,
            "All measurable outputs (theoretical maximum observability)",
            [
                log_HBsAg ~ log10(ode_system.S), log_V ~ log10(ode_system.V),
                log_ALT ~ log10(ode_system.Z),
            ],
            clinically_relevant = false
        )
    )
end

#=============================================================================
# Helper Functions
=============================================================================#

"""
    extract_ode_system(pumas_model)

Extract the underlying ModelingToolkit ODESystem from a Pumas model.

# Arguments
- `pumas_model`: A Pumas @model definition

# Returns
ModelingToolkit ODESystem
"""
function extract_ode_system(pumas_model)
    return pumas_model.sys
end

"""
    get_ode_parameters(ode_system)

Get all parameters from an ODESystem.

# Arguments
- `ode_system`: ModelingToolkit ODESystem

# Returns
Vector of parameter symbols
"""
function get_ode_parameters(ode_system)
    return [Symbol(string(p)) for p in ode_system.ps]
end

"""
    get_ode_states(ode_system)

Get all state variables from an ODESystem.

# Arguments
- `ode_system`: ModelingToolkit ODESystem

# Returns
Vector of state variable symbols
"""
function get_ode_states(ode_system)
    return [Symbol(string(s)) for s in ode_system.states]
end

#=============================================================================
# Core Analysis Functions
=============================================================================#

"""
    assess_scenario_identifiability(
        ode_system,
        scenario::MeasurementScenario;
        funcs_to_check::Vector = [],
        prob_threshold::Float64 = 0.99,
        verbose::Bool = true
    ) -> IdentifiabilityResult

Assess global structural identifiability for a measurement scenario.

# Arguments
- `ode_system`: ModelingToolkit ODESystem
- `scenario`: MeasurementScenario defining what can be measured
- `funcs_to_check`: Specific functions/parameters to check (empty = all)
- `prob_threshold`: Probability threshold for correctness (default: 0.99)
- `verbose`: Print progress information

# Returns
IdentifiabilityResult with identifiability classifications

# Example
```julia
ode_sys = extract_ode_system(tumor_burden_model)
scenarios = create_tumor_burden_scenarios(ode_sys)
result = assess_scenario_identifiability(ode_sys, scenarios[:tumor_size_only])
```
"""
function assess_scenario_identifiability(
        ode_system,
        scenario::MeasurementScenario;
        funcs_to_check::Vector = [],
        prob_threshold::Float64 = 0.99,
        verbose::Bool = true
    )
    if verbose
        println("\n" * "="^60)
        println("Scenario: $(scenario.name)")
        println("Description: $(scenario.description)")
        println("Clinically relevant: $(scenario.clinically_relevant)")
        println("="^60)
    end

    # Assess identifiability
    ident_result = assess_identifiability(
        ode_system,
        measured_quantities = scenario.measured_quantities,
        funcs_to_check = funcs_to_check,
        prob_threshold = prob_threshold
    )

    if verbose
        println("\nIdentifiability Results:")
        println("-"^40)
        for (param, status) in sort(collect(ident_result), by = x -> string(x[1]))
            status_str = status == :globally ? "globally identifiable" :
                (status == :locally ? "locally identifiable" : "non-identifiable")
            println("  $(param) => $(status_str)")
        end
    end

    result = IdentifiabilityResult(scenario, ident_result)

    if verbose
        println(
            "\nSummary: ", result.all_globally_identifiable ?
                "All parameters are GLOBALLY IDENTIFIABLE" :
                "Some parameters are NOT globally identifiable"
        )
    end

    return result
end

"""
    assess_local_scenario_identifiability(
        ode_system,
        scenario::MeasurementScenario;
        funcs_to_check::Vector = [],
        type::Symbol = :SE,
        prob_threshold::Float64 = 0.99,
        verbose::Bool = true
    ) -> LocalIdentifiabilityResult

Assess local structural identifiability for a measurement scenario.
Local identifiability analysis is faster than global and may be sufficient
for complex models like HBV.

# Arguments
- `ode_system`: ModelingToolkit ODESystem
- `scenario`: MeasurementScenario defining what can be measured
- `funcs_to_check`: Specific functions/parameters to check (empty = all)
- `type`: :SE for single experiment, :ME for multi-experiment
- `prob_threshold`: Probability threshold for correctness
- `verbose`: Print progress information

# Returns
LocalIdentifiabilityResult with local identifiability classifications
"""
function assess_local_scenario_identifiability(
        ode_system,
        scenario::MeasurementScenario;
        funcs_to_check::Vector = [],
        type::Symbol = :SE,
        prob_threshold::Float64 = 0.99,
        verbose::Bool = true
    )
    if verbose
        println("\n" * "="^60)
        println("Local Identifiability Analysis")
        println("Scenario: $(scenario.name)")
        println("Type: $(type == :SE ? "Single Experiment" : "Multi-Experiment")")
        println("="^60)
    end

    # Assess local identifiability
    ident_result = assess_local_identifiability(
        ode_system,
        measured_quantities = scenario.measured_quantities,
        funcs_to_check = funcs_to_check,
        prob_threshold = prob_threshold,
        type = type
    )

    if verbose
        println("\nLocal Identifiability Results:")
        println("-"^40)
        for (param, is_id) in sort(collect(ident_result), by = x -> string(x[1]))
            status_str = is_id ? "locally identifiable" : "non-identifiable"
            println("  $(param) => $(status_str)")
        end
    end

    result = LocalIdentifiabilityResult(
        scenario, ident_result;
        type = type,
        num_experiments = type == :SE ? 1 : 2
    )

    if verbose
        n_id = length(result.identifiable)
        n_total = n_id + length(result.nonidentifiable)
        println("\nSummary: $(n_id)/$(n_total) parameters are locally identifiable")
    end

    return result
end

"""
    find_scenario_identifiable_functions(
        ode_system,
        scenario::MeasurementScenario;
        with_states::Bool = false,
        simplify::Symbol = :standard,
        verbose::Bool = true
    ) -> IdentifiableFunctionsResult

Find identifiable parameter combinations for a measurement scenario.

When individual parameters are not identifiable, this function identifies
what combinations of parameters ARE identifiable.

# Arguments
- `ode_system`: ModelingToolkit ODESystem
- `scenario`: MeasurementScenario defining what can be measured
- `with_states`: Include state variables in the analysis
- `simplify`: Simplification level (:standard, :weak, or :strong)
- `verbose`: Print progress information

# Returns
IdentifiableFunctionsResult with identifiable parameter combinations
"""
function find_scenario_identifiable_functions(
        ode_system,
        scenario::MeasurementScenario;
        with_states::Bool = false,
        simplify::Symbol = :standard,
        verbose::Bool = true
    )
    if verbose
        println("\n" * "="^60)
        println("Finding Identifiable Functions")
        println("Scenario: $(scenario.name)")
        println("="^60)
    end

    # Convert to SI format
    si_ode, si_dict = StructuralIdentifiability.mtk_to_si(
        ode_system, scenario.measured_quantities
    )

    # Find identifiable functions
    funcs = find_identifiable_functions(
        si_ode,
        with_states = with_states,
        simplify = simplify
    )

    if verbose
        println("\nIdentifiable Functions:")
        println("-"^40)
        for (i, func) in enumerate(funcs)
            println("  $i. $func")
        end
    end

    return IdentifiableFunctionsResult(scenario, funcs, with_states, simplify == :standard)
end

"""
    compute_scenario_reparameterization(
        ode_system,
        scenario::MeasurementScenario;
        verbose::Bool = true
    ) -> ReparameterizationResult

Compute a globally identifiable reparameterization of the model.

This suggests how to rewrite the model with identifiable parameter combinations.

# Arguments
- `ode_system`: ModelingToolkit ODESystem
- `scenario`: MeasurementScenario defining what can be measured
- `verbose`: Print progress information

# Returns
ReparameterizationResult with new variable mappings and recommendations
"""
function compute_scenario_reparameterization(
        ode_system,
        scenario::MeasurementScenario;
        verbose::Bool = true
    )
    if verbose
        println("\n" * "="^60)
        println("Computing Reparameterization")
        println("Scenario: $(scenario.name)")
        println("="^60)
    end

    # Convert to SI format
    si_ode, si_dict = StructuralIdentifiability.mtk_to_si(
        ode_system, scenario.measured_quantities
    )

    # Compute reparameterization
    reparam = reparameterize_global(si_ode)

    # Generate recommendations
    recommendations = String[]

    if verbose
        println("\nReparameterization (identifiable parameter combinations):")
        println("-"^40)

        println("  New -> Original mapping:")
        for (new_var, orig_expr) in sort(collect(reparam[:new_vars]), by = x -> string(x[1]))
            var_str = string(new_var)
            if startswith(var_str, "a")  # Parameters are named a1, a2, etc.
                println("    $new_var = $orig_expr")
                push!(recommendations, "Consider estimating $new_var = $orig_expr")
            end
        end
    end

    return ReparameterizationResult(scenario, reparam[:new_vars], recommendations)
end

#=============================================================================
# High-Level Analysis Functions
=============================================================================#

"""
    analyze_tumor_burden_identifiability(;
        scenarios::Vector{Symbol} = [:tumor_size_only, :both_populations],
        include_global::Bool = true,
        include_local::Bool = true,
        find_functions::Bool = true,
        compute_reparameterization::Bool = true,
        prob_threshold::Float64 = 0.99,
        verbose::Bool = true
    ) -> ComprehensiveIdentifiabilityReport

Perform comprehensive identifiability analysis on the tumor burden model.

# Arguments
- `scenarios`: Which measurement scenarios to analyze
- `include_global`: Run global identifiability analysis
- `include_local`: Run local identifiability analysis
- `find_functions`: Find identifiable parameter combinations
- `compute_reparameterization`: Compute identifiable reparameterization
- `prob_threshold`: Probability threshold for correctness
- `verbose`: Print progress information

# Returns
ComprehensiveIdentifiabilityReport with all analysis results

# Example
```julia
report = analyze_tumor_burden_identifiability()
print_identifiability_report(report)
```
"""
function analyze_tumor_burden_identifiability(;
        scenarios::Vector{Symbol} = [:tumor_size_only, :both_populations],
        include_global::Bool = true,
        include_local::Bool = true,
        find_functions::Bool = true,
        compute_reparameterization::Bool = true,
        prob_threshold::Float64 = 0.99,
        verbose::Bool = true
    )
    if verbose
        println("\n" * "="^70)
        println("  TUMOR BURDEN MODEL STRUCTURAL IDENTIFIABILITY ANALYSIS")
        println("="^70)
    end

    # Extract ODE system from the tumor burden model
    ode_system = extract_ode_system(tumor_burden_model)

    # Create scenarios
    available_scenarios = create_tumor_burden_scenarios(ode_system)

    # Initialize result containers
    global_results = Dict{Symbol, IdentifiabilityResult}()
    local_results = Dict{Symbol, LocalIdentifiabilityResult}()
    functions_results = Dict{Symbol, IdentifiableFunctionsResult}()
    reparameterization_results = Dict{Symbol, ReparameterizationResult}()

    for scenario_name in scenarios
        if !haskey(available_scenarios, scenario_name)
            @warn "Scenario $scenario_name not found, skipping"
            continue
        end

        scenario = available_scenarios[scenario_name]

        # Global analysis
        if include_global
            global_results[scenario_name] = assess_scenario_identifiability(
                ode_system, scenario;
                prob_threshold = prob_threshold,
                verbose = verbose
            )
        end

        # Local analysis
        if include_local
            local_results[scenario_name] = assess_local_scenario_identifiability(
                ode_system, scenario;
                prob_threshold = prob_threshold,
                verbose = verbose
            )
        end

        # Find identifiable functions
        if find_functions
            functions_results[scenario_name] = find_scenario_identifiable_functions(
                ode_system, scenario;
                verbose = verbose
            )
        end

        # Compute reparameterization
        if compute_reparameterization
            reparameterization_results[scenario_name] = compute_scenario_reparameterization(
                ode_system, scenario;
                verbose = verbose
            )
        end
    end

    # Create summary DataFrame
    summary = compare_scenarios(global_results)

    return ComprehensiveIdentifiabilityReport(
        "Tumor Burden Model",
        global_results,
        local_results,
        functions_results,
        reparameterization_results,
        summary
    )
end

"""
    analyze_hbv_identifiability(;
        scenarios::Vector{Symbol} = [:hbsag_and_viral],
        params_to_check::Vector{Symbol} = HBV_ESTIMATED_PARAM_NAMES,
        local_type::Symbol = :SE,
        include_global::Bool = false,
        include_local::Bool = true,
        find_functions::Bool = false,
        compute_reparameterization::Bool = false,
        prob_threshold::Float64 = 0.99,
        verbose::Bool = true
    ) -> ComprehensiveIdentifiabilityReport

Perform identifiability analysis on the HBV QSP model.

Due to the model complexity (11 ODEs, 31 parameters), this defaults to local
identifiability analysis which is faster. Global analysis can be enabled but
may take significant time.

# Arguments
- `scenarios`: Which measurement scenarios to analyze
- `params_to_check`: Which parameters to check (default: 9 estimated params)
- `local_type`: :SE for single experiment, :ME for multi-experiment
- `include_global`: Run global identifiability (slow for HBV model)
- `include_local`: Run local identifiability (faster)
- `find_functions`: Find identifiable parameter combinations
- `compute_reparameterization`: Compute identifiable reparameterization
- `prob_threshold`: Probability threshold for correctness
- `verbose`: Print progress information

# Returns
ComprehensiveIdentifiabilityReport with all analysis results

# Example
```julia
# Quick local analysis
report = analyze_hbv_identifiability(scenarios=[:hbsag_and_viral])

# Full analysis (slower)
report = analyze_hbv_identifiability(
    include_global=true,
    find_functions=true
)
```
"""
function analyze_hbv_identifiability(;
        scenarios::Vector{Symbol} = [:hbsag_and_viral],
        params_to_check::Vector{Symbol} = HBV_ESTIMATED_PARAM_NAMES,
        local_type::Symbol = :SE,
        include_global::Bool = false,
        include_local::Bool = true,
        find_functions::Bool = false,
        compute_reparameterization::Bool = false,
        prob_threshold::Float64 = 0.99,
        verbose::Bool = true
    )
    if verbose
        println("\n" * "="^70)
        println("  HBV QSP MODEL STRUCTURAL IDENTIFIABILITY ANALYSIS")
        println("  Note: HBV model is complex (11 ODEs, 31 params)")
        println("  Default: local analysis on 9 estimated parameters")
        println("="^70)
    end

    # Extract ODE system from the HBV model
    ode_system = extract_ode_system(hbv_model)

    # Create scenarios
    available_scenarios = create_hbv_scenarios(ode_system)

    # Initialize result containers
    global_results = Dict{Symbol, IdentifiabilityResult}()
    local_results = Dict{Symbol, LocalIdentifiabilityResult}()
    functions_results = Dict{Symbol, IdentifiableFunctionsResult}()
    reparameterization_results = Dict{Symbol, ReparameterizationResult}()

    for scenario_name in scenarios
        if !haskey(available_scenarios, scenario_name)
            @warn "Scenario $scenario_name not found, skipping"
            continue
        end

        scenario = available_scenarios[scenario_name]

        # Global analysis (slow for HBV)
        if include_global
            if verbose
                println("\nNote: Global analysis may take significant time for HBV model...")
            end
            global_results[scenario_name] = assess_scenario_identifiability(
                ode_system, scenario;
                prob_threshold = prob_threshold,
                verbose = verbose
            )
        end

        # Local analysis (faster)
        if include_local
            local_results[scenario_name] = assess_local_scenario_identifiability(
                ode_system, scenario;
                type = local_type,
                prob_threshold = prob_threshold,
                verbose = verbose
            )
        end

        # Find identifiable functions
        if find_functions
            functions_results[scenario_name] = find_scenario_identifiable_functions(
                ode_system, scenario;
                verbose = verbose
            )
        end

        # Compute reparameterization
        if compute_reparameterization
            reparameterization_results[scenario_name] = compute_scenario_reparameterization(
                ode_system, scenario;
                verbose = verbose
            )
        end
    end

    # Create summary DataFrame
    summary = if !isempty(global_results)
        compare_scenarios(global_results)
    else
        compare_local_scenarios(local_results)
    end

    return ComprehensiveIdentifiabilityReport(
        "HBV QSP Model",
        global_results,
        local_results,
        functions_results,
        reparameterization_results,
        summary
    )
end

#=============================================================================
# Summary and Comparison Functions
=============================================================================#

"""
    compare_scenarios(results::Dict{Symbol, IdentifiabilityResult}) -> DataFrame

Create a comparison table across measurement scenarios.

# Returns
DataFrame with columns: scenario, all_global, n_global, n_local, n_nonid, clinical
"""
function compare_scenarios(results::Dict{Symbol, IdentifiabilityResult})
    rows = NamedTuple[]

    for (name, result) in sort(collect(results), by = x -> string(x[1]))
        push!(
            rows, (
                scenario = name,
                all_global = result.all_globally_identifiable,
                n_global = length(result.globally_identifiable),
                n_local = length(result.locally_identifiable),
                n_nonid = length(result.nonidentifiable),
                clinical = result.scenario.clinically_relevant,
            )
        )
    end

    return DataFrame(rows)
end

"""
    compare_local_scenarios(results::Dict{Symbol, LocalIdentifiabilityResult}) -> DataFrame

Create a comparison table for local identifiability results.

# Returns
DataFrame with columns: scenario, n_identifiable, n_nonid, type, clinical
"""
function compare_local_scenarios(results::Dict{Symbol, LocalIdentifiabilityResult})
    rows = NamedTuple[]

    for (name, result) in sort(collect(results), by = x -> string(x[1]))
        push!(
            rows, (
                scenario = name,
                n_identifiable = length(result.identifiable),
                n_nonid = length(result.nonidentifiable),
                type = result.type,
                clinical = result.scenario.clinically_relevant,
            )
        )
    end

    return DataFrame(rows)
end

"""
    summarize_identifiability(results::Dict{Symbol, IdentifiabilityResult}) -> DataFrame

Create detailed summary of identifiability results across scenarios.

# Returns
DataFrame with parameter-level identifiability status for each scenario
"""
function summarize_identifiability(results::Dict{Symbol, IdentifiabilityResult})
    # Get all unique parameters
    all_params = Set{Symbol}()
    for result in values(results)
        for param in keys(result.identifiability)
            push!(all_params, Symbol(string(param)))
        end
    end
    all_params = sort(collect(all_params))

    # Build table
    rows = NamedTuple[]
    for param in all_params
        row = Dict{Symbol, Any}(:parameter => param)
        for (scenario_name, result) in results
            # Find matching parameter
            status = :not_found
            for (p, s) in result.identifiability
                if Symbol(string(p)) == param
                    status = s
                    break
                end
            end
            row[scenario_name] = status
        end
        push!(rows, NamedTuple(row))
    end

    return DataFrame(rows)
end

"""
    generate_recommendations(
        results::Dict{Symbol, IdentifiabilityResult},
        reparam::Union{ReparameterizationResult, Nothing} = nothing
    ) -> Vector{String}

Generate practical recommendations based on identifiability analysis.

# Arguments
- `results`: Global identifiability results
- `reparam`: Optional reparameterization results

# Returns
Vector of recommendation strings
"""
function generate_recommendations(
        results::Dict{Symbol, IdentifiabilityResult};
        reparam::Union{ReparameterizationResult, Nothing} = nothing
    )
    recommendations = String[]

    # Find best clinically relevant scenario
    best_clinical = nothing
    best_n_global = 0

    for (name, result) in results
        if result.scenario.clinically_relevant
            n_global = length(result.globally_identifiable)
            if n_global > best_n_global
                best_n_global = n_global
                best_clinical = result
            end
        end
    end

    if !isnothing(best_clinical)
        if best_clinical.all_globally_identifiable
            push!(
                recommendations,
                "GOOD NEWS: All parameters are globally identifiable with $(best_clinical.scenario.name) measurements"
            )
        else
            push!(
                recommendations,
                "RECOMMENDATION: Use $(best_clinical.scenario.name) scenario for best clinical identifiability"
            )

            if !isempty(best_clinical.nonidentifiable)
                push!(
                    recommendations,
                    "WARNING: Parameters $(join(best_clinical.nonidentifiable, ", ")) are non-identifiable"
                )
                push!(
                    recommendations,
                    "Consider: (1) Adding measurements, (2) Fixing to literature values, or (3) Using informative priors"
                )
            end
        end
    end

    # Add reparameterization recommendations
    if !isnothing(reparam) && !isempty(reparam.recommendations)
        push!(recommendations, "REPARAMETERIZATION SUGGESTIONS:")
        append!(recommendations, reparam.recommendations)
    end

    return recommendations
end

#=============================================================================
# Printing Functions
=============================================================================#

"""
    print_identifiability_result(result::IdentifiabilityResult; verbose::Bool = true)

Print formatted identifiability result.
"""
function print_identifiability_result(result::IdentifiabilityResult; verbose::Bool = true)
    println("\n" * "="^60)
    println("Scenario: $(result.scenario.name)")
    println("Description: $(result.scenario.description)")
    println("="^60)

    println("\nGlobally Identifiable ($(length(result.globally_identifiable))):")
    for p in result.globally_identifiable
        println("  - $p")
    end

    if !isempty(result.locally_identifiable)
        println("\nLocally Identifiable ($(length(result.locally_identifiable))):")
        for p in result.locally_identifiable
            println("  - $p")
        end
    end

    if !isempty(result.nonidentifiable)
        println("\nNon-Identifiable ($(length(result.nonidentifiable))):")
        for p in result.nonidentifiable
            println("  - $p")
        end
    end

    status = result.all_globally_identifiable ?
        "ALL GLOBALLY IDENTIFIABLE" : "SOME PARAMETERS NOT IDENTIFIABLE"
    return println("\nStatus: $status")
end

"""
    print_identifiability_report(report::ComprehensiveIdentifiabilityReport)

Print comprehensive identifiability report.
"""
function print_identifiability_report(report::ComprehensiveIdentifiabilityReport)
    println("\n" * "="^70)
    println("  COMPREHENSIVE IDENTIFIABILITY REPORT")
    println("  Model: $(report.model_name)")
    println("="^70)

    # Print summary table
    if nrow(report.summary) > 0
        println("\nSCENARIO COMPARISON:")
        println("-"^60)
        println(report.summary)
    end

    # Print global results
    if !isempty(report.global_results)
        println("\n\nGLOBAL IDENTIFIABILITY RESULTS:")
        for (name, result) in report.global_results
            print_identifiability_result(result)
        end
    end

    # Print local results
    if !isempty(report.local_results)
        println("\n\nLOCAL IDENTIFIABILITY RESULTS:")
        for (name, result) in report.local_results
            println("\nScenario: $name ($(result.type))")
            println("  Identifiable: $(join(result.identifiable, ", "))")
            if !isempty(result.nonidentifiable)
                println("  Non-identifiable: $(join(result.nonidentifiable, ", "))")
            end
        end
    end

    # Print recommendations
    return if !isempty(report.global_results)
        recommendations = generate_recommendations(report.global_results)
        if !isempty(recommendations)
            println("\n\nRECOMMENDATIONS:")
            println("-"^60)
            for rec in recommendations
                println("  - $rec")
            end
        end
    end
end

#=============================================================================
# Exports
=============================================================================#

export MeasurementScenario, IdentifiabilityResult, LocalIdentifiabilityResult
export IdentifiableFunctionsResult, ReparameterizationResult, ComprehensiveIdentifiabilityReport

export HBV_ESTIMATED_PARAM_NAMES
export create_tumor_burden_scenarios, create_hbv_scenarios

export extract_ode_system, get_ode_parameters, get_ode_states

export assess_scenario_identifiability, assess_local_scenario_identifiability
export find_scenario_identifiable_functions, compute_scenario_reparameterization

export analyze_tumor_burden_identifiability, analyze_hbv_identifiability

export compare_scenarios, compare_local_scenarios, summarize_identifiability
export generate_recommendations

export print_identifiability_result, print_identifiability_report
