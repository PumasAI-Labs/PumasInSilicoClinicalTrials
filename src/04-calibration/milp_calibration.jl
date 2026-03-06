"""
    MILP-Based Virtual Population Calibration

This module implements Step 5 of the ISCT workflow (Figure 3 in paper):
"Select Virtual Patients and Calibrate Vpop"

The algorithm uses Mixed-Integer Linear Programming (MILP) to select a subset
of virtual patients whose distribution matches a target clinical distribution.

Key concepts:
1. Maximize the number of selected virtual patients
2. Ensure distribution matching within tolerance (epsilon)
3. Use multi-objective optimization to find optimal (nbins, epsilon) Pareto front

Reference:
    Section 2, Step 5 of the ISCT workflow paper (Figure 3)
"""

using JuMP
using HiGHS
using DataFrames
using Statistics
using StatsBase: fit, Histogram

#=============================================================================
# Data Structures
=============================================================================#

"""
    TargetDistribution

Specification of a target clinical distribution for calibration.

# Fields
- `edges::Vector{Float64}`: Bin edges (n+1 values for n bins)
- `percentages::Vector{Float64}`: Target percentages for each bin (should sum to ~100)
- `variable_name::Symbol`: Name of the variable being matched
"""
struct TargetDistribution
    edges::Vector{Float64}
    percentages::Vector{Float64}
    variable_name::Symbol

    function TargetDistribution(edges, percentages, variable_name = :value)
        @assert length(edges) == length(percentages) + 1 "Edges should have one more element than percentages"
        @assert all(percentages .>= 0) "Percentages must be non-negative"
        return new(edges, percentages, variable_name)
    end
end

"""
    CalibrationResult

Results from MILP calibration.

# Fields
- `selected_indices::Vector{Int}`: Indices of selected virtual patients
- `n_selected::Int`: Total number of selected patients
- `n_original::Int`: Original number of patients
- `selection_rate::Float64`: Fraction of patients selected
- `mean_error::Float64`: Mean distribution matching error (%)
- `bin_errors::Vector{Float64}`: Per-bin errors
- `solver_status::Symbol`: Optimization solver status
"""
struct CalibrationResult
    selected_indices::Vector{Int}
    n_selected::Int
    n_original::Int
    selection_rate::Float64
    mean_error::Float64
    bin_errors::Vector{Float64}
    solver_status::Symbol
end

#=============================================================================
# Target Distribution Creation
=============================================================================#

"""
    create_target_from_data(
        data::AbstractVector;
        nbins::Int = 20,
        variable_name::Symbol = :value
    ) -> TargetDistribution

Create a target distribution from empirical data using histogram binning.

# Example
```julia
clinical_data = randn(1000) .* 0.5 .+ log10(1250)
target = create_target_from_data(clinical_data; nbins=20)
```
"""
function create_target_from_data(
        data::AbstractVector;
        nbins::Int = 20,
        variable_name::Symbol = :value
    )
    h = fit(Histogram, data, nbins = nbins)
    edges = collect(h.edges[1])
    counts = h.weights
    percentages = 100.0 .* counts ./ sum(counts)

    return TargetDistribution(edges, percentages, variable_name)
end

"""
    create_target_from_specification(
        edges::Vector{<:Real},
        percentages::Vector{<:Real};
        variable_name::Symbol = :value
    ) -> TargetDistribution

Create a target distribution from pre-specified bin edges and percentages.

This is useful for matching to published clinical trial distributions.

# Example (Everest trial HBsAg distribution)
```julia
edges = [log10(0.03), log10(0.05), log10(100), log10(200),
         log10(500), log10(1000), log10(1500), log10(10000)]
percentages = [0, 37.5, 10.9, 19.1, 21.5, 11, 0]
target = create_target_from_specification(edges, percentages; variable_name=:HBsAg)
```
"""
function create_target_from_specification(
        edges::Vector{<:Real},
        percentages::Vector{<:Real};
        variable_name::Symbol = :value
    )
    return TargetDistribution(Float64.(edges), Float64.(percentages), variable_name)
end

#=============================================================================
# VP Classification
=============================================================================#

"""
    classify_vps_to_bins(
        vp_values::AbstractVector,
        edges::Vector{Float64}
    ) -> Tuple{Vector{Int}, BitVector}

Classify virtual patients into bins based on their values.

Returns:
- `bin_assignments`: Bin index for each VP (0 if outside all bins)
- `in_range`: BitVector indicating which VPs are within the distribution range
"""
function classify_vps_to_bins(
        vp_values::AbstractVector,
        edges::Vector{Float64}
    )
    n_vps = length(vp_values)
    n_bins = length(edges) - 1

    bin_assignments = zeros(Int, n_vps)
    in_range = falses(n_vps)

    for i in 1:n_vps
        val = vp_values[i]
        for b in 1:n_bins
            if val > edges[b] && val <= edges[b + 1]
                bin_assignments[i] = b
                in_range[i] = true
                break
            end
        end
    end

    return bin_assignments, in_range
end

"""
    create_bin_membership_matrix(
        bin_assignments::Vector{Int},
        n_bins::Int;
        only_in_range::Bool = true,
        in_range::BitVector = trues(length(bin_assignments))
    ) -> Tuple{Matrix{Float64}, Vector{Int}}

Create a binary matrix indicating VP membership in each bin.

Returns:
- `membership_matrix`: (n_bins × n_filtered_vps) binary matrix
- `original_indices`: Original indices of VPs in the filtered set
"""
function create_bin_membership_matrix(
        bin_assignments::Vector{Int},
        n_bins::Int;
        only_in_range::Bool = true,
        in_range::BitVector = trues(length(bin_assignments))
    )
    if only_in_range
        # Filter to only VPs within range
        filtered_indices = findall(in_range)
        filtered_assignments = bin_assignments[in_range]
    else
        filtered_indices = collect(1:length(bin_assignments))
        filtered_assignments = bin_assignments
    end

    n_filtered = length(filtered_indices)
    membership = zeros(Float64, n_bins, n_filtered)

    for (j, bin) in enumerate(filtered_assignments)
        if bin > 0
            membership[bin, j] = 1.0
        end
    end

    return membership, filtered_indices
end

#=============================================================================
# MILP Solver
=============================================================================#

"""
    solve_milp_calibration(
        vp_values::AbstractVector,
        target::TargetDistribution,
        epsilon::Float64;
        time_limit::Float64 = 60.0,
        silent::Bool = true
    ) -> CalibrationResult

Solve the MILP calibration problem.

# Problem Formulation

**Decision Variables:**
- `n_total`: Integer, total number of selected VPs
- `x[j]`: Binary, 1 if VP j is selected, 0 otherwise

**Objective:**
Maximize `n_total` (maximize number of selected VPs)

**Constraints:**
For each bin b with target fraction `p_b`:
- `p_b * n_total - sum(x[j] for j in bin_b) <= epsilon * p_b * n_total`
- `-p_b * n_total + sum(x[j] for j in bin_b) <= epsilon * p_b * n_total`

**Equality Constraint:**
- `n_total = sum(x[j] for all j)`

# Arguments
- `vp_values`: Vector of VP values for the variable being calibrated
- `target`: Target distribution specification
- `epsilon`: Tolerance for distribution matching (0.1 = 10% error allowed)
- `time_limit`: Maximum solver time in seconds
- `silent`: Whether to suppress solver output

# Returns
CalibrationResult with selected VP indices and diagnostics
"""
function solve_milp_calibration(
        vp_values::AbstractVector,
        target::TargetDistribution,
        epsilon::Float64;
        time_limit::Float64 = 60.0,
        silent::Bool = true
    )
    n_vps = length(vp_values)
    n_bins = length(target.percentages)

    # Target fractions (convert from percentages)
    target_fractions = target.percentages ./ 100.0

    # Classify VPs to bins
    bin_assignments, in_range = classify_vps_to_bins(vp_values, target.edges)

    # Create membership matrix (only for VPs within range)
    membership, filtered_indices = create_bin_membership_matrix(
        bin_assignments, n_bins; only_in_range = true, in_range = in_range
    )

    n_filtered = length(filtered_indices)

    if n_filtered == 0
        return CalibrationResult(
            Int[], 0, n_vps, 0.0, Inf, Float64[], :NO_FEASIBLE_VPS
        )
    end

    # Create optimization model
    model = Model(HiGHS.Optimizer)
    if silent
        set_silent(model)
    end
    set_time_limit_sec(model, time_limit)

    # Decision variables
    @variable(model, n_total >= 1, Int)  # Total selected VPs
    @variable(model, x[1:n_filtered], Bin)  # Binary selection for each VP

    # Objective: maximize total selected VPs
    @objective(model, Max, n_total)

    # Equality constraint: n_total = sum of selected
    @constraint(model, n_total == sum(x))

    # Distribution matching constraints for each bin
    eps_scaled = epsilon * 100  # Scale to match MATLAB implementation

    for b in 1:n_bins
        p_b = round(target_fractions[b], digits = 3)
        if p_b > 0  # Only add constraints for non-empty bins
            # VP count in bin b
            bin_count = sum(membership[b, j] * x[j] for j in 1:n_filtered)

            # Upper bound: bin_count <= p_b * n_total * (1 + epsilon)
            @constraint(model, p_b * n_total - bin_count <= eps_scaled * p_b * n_total / 100)

            # Lower bound: bin_count >= p_b * n_total * (1 - epsilon)
            @constraint(model, -p_b * n_total + bin_count <= eps_scaled * p_b * n_total / 100)
        end
    end

    # Bounds on n_total
    @constraint(model, n_total >= n_bins)  # At least nbins VPs
    @constraint(model, n_total <= n_filtered)  # At most all filtered VPs

    # Solve
    optimize!(model)

    # Extract results
    status = termination_status(model)

    if status == MOI.OPTIMAL || status == MOI.FEASIBLE_POINT || status == MOI.TIME_LIMIT
        x_vals = value.(x)
        n_selected = round(Int, value(n_total))

        # Get selected VP indices (map back to original indices)
        selected_filtered = findall(x_vals .> 0.5)
        selected_indices = filtered_indices[selected_filtered]

        # Calculate bin errors
        selected_membership = membership[:, selected_filtered]
        actual_fractions = vec(sum(selected_membership, dims = 2)) ./ n_selected
        bin_errors = 100.0 .* abs.(actual_fractions .- target_fractions)
        mean_error = mean(bin_errors)

        return CalibrationResult(
            selected_indices,
            n_selected,
            n_vps,
            n_selected / n_vps,
            mean_error,
            bin_errors,
            Symbol(status)
        )
    else
        return CalibrationResult(
            Int[], 0, n_vps, 0.0, Inf, Float64[], Symbol(status)
        )
    end
end

#=============================================================================
# Multi-Variable Calibration
=============================================================================#

"""
    solve_multivariable_calibration(
        vp_data::DataFrame,
        targets::Vector{TargetDistribution},
        epsilon::Float64;
        time_limit::Float64 = 120.0,
        silent::Bool = true
    ) -> CalibrationResult

Calibrate VP selection to match multiple target distributions simultaneously.

# Arguments
- `vp_data`: DataFrame with columns matching target variable names
- `targets`: Vector of target distributions for each variable
- `epsilon`: Tolerance for distribution matching
- `time_limit`: Maximum solver time
- `silent`: Whether to suppress solver output

# Returns
CalibrationResult selecting VPs that match all target distributions
"""
function solve_multivariable_calibration(
        vp_data::DataFrame,
        targets::Vector{TargetDistribution},
        epsilon::Float64;
        time_limit::Float64 = 120.0,
        silent::Bool = true
    )
    n_vps = nrow(vp_data)
    n_vars = length(targets)

    # Compute total bins and target fractions
    total_bins = sum(length(t.percentages) for t in targets)

    # Classify VPs for each variable and find VPs in range for ALL variables
    all_in_range = trues(n_vps)
    bin_assignments_all = Vector{Vector{Int}}()
    target_fractions_all = Vector{Float64}()

    for target in targets
        var_name = target.variable_name
        @assert hasproperty(vp_data, var_name) "VP data missing column: $var_name"

        vp_values = vp_data[!, var_name]
        bin_assignments, in_range = classify_vps_to_bins(vp_values, target.edges)

        all_in_range .&= in_range
        push!(bin_assignments_all, bin_assignments)
        append!(target_fractions_all, target.percentages ./ 100.0)
    end

    # Filter to VPs in range for all variables
    filtered_indices = findall(all_in_range)
    n_filtered = length(filtered_indices)

    if n_filtered == 0
        return CalibrationResult(
            Int[], 0, n_vps, 0.0, Inf, Float64[], :NO_FEASIBLE_VPS
        )
    end

    # Create combined membership matrix
    membership = zeros(Float64, total_bins, n_filtered)
    bin_offset = 0

    for (v, target) in enumerate(targets)
        n_bins_v = length(target.percentages)
        bin_assignments = bin_assignments_all[v]

        for (j, orig_idx) in enumerate(filtered_indices)
            bin = bin_assignments[orig_idx]
            if bin > 0
                membership[bin_offset + bin, j] = 1.0
            end
        end

        bin_offset += n_bins_v
    end

    # Create optimization model
    model = Model(HiGHS.Optimizer)
    if silent
        set_silent(model)
    end
    set_time_limit_sec(model, time_limit)

    # Decision variables
    @variable(model, n_total >= 1, Int)
    @variable(model, x[1:n_filtered], Bin)

    # Objective: maximize total selected
    @objective(model, Max, n_total)

    # Equality constraint
    @constraint(model, n_total == sum(x))

    # Distribution matching constraints
    eps_scaled = epsilon * 100

    for b in 1:total_bins
        p_b = round(target_fractions_all[b], digits = 3)
        if p_b > 0
            bin_count = sum(membership[b, j] * x[j] for j in 1:n_filtered)
            @constraint(model, p_b * n_total - bin_count <= eps_scaled * p_b * n_total / 100)
            @constraint(model, -p_b * n_total + bin_count <= eps_scaled * p_b * n_total / 100)
        end
    end

    # Bounds
    @constraint(model, n_total >= total_bins)
    @constraint(model, n_total <= n_filtered)

    # Solve
    optimize!(model)

    # Extract results
    status = termination_status(model)

    if status == MOI.OPTIMAL || status == MOI.FEASIBLE_POINT || status == MOI.TIME_LIMIT
        x_vals = value.(x)
        n_selected = round(Int, value(n_total))

        selected_filtered = findall(x_vals .> 0.5)
        selected_indices = filtered_indices[selected_filtered]

        # Calculate errors
        selected_membership = membership[:, selected_filtered]
        actual_fractions = vec(sum(selected_membership, dims = 2)) ./ n_selected
        bin_errors = 100.0 .* abs.(actual_fractions .- target_fractions_all)
        mean_error = mean(bin_errors)

        return CalibrationResult(
            selected_indices,
            n_selected,
            n_vps,
            n_selected / n_vps,
            mean_error,
            bin_errors,
            Symbol(status)
        )
    else
        return CalibrationResult(
            Int[], 0, n_vps, 0.0, Inf, Float64[], Symbol(status)
        )
    end
end

#=============================================================================
# Multi-Objective Optimization
=============================================================================#

"""
    ParetoPoint

A point on the Pareto front from multi-objective optimization.
"""
struct ParetoPoint
    nbins::Int
    epsilon::Float64
    n_selected::Int
    mean_error::Float64
    result::CalibrationResult
end

"""
    find_pareto_front(
        vp_values::AbstractVector,
        target_data::AbstractVector;
        nbins_range::UnitRange{Int} = 10:50,
        epsilon_range::Tuple{Float64, Float64} = (0.01, 0.5),
        n_epsilon_samples::Int = 20,
        time_limit_per_solve::Float64 = 10.0,
        silent::Bool = true
    ) -> Vector{ParetoPoint}

Find the Pareto front of (nbins, epsilon) combinations.

This is a grid search approach that evaluates multiple combinations and
returns the non-dominated solutions.

# Arguments
- `vp_values`: Virtual patient values for the variable
- `target_data`: Clinical data to match distribution against
- `nbins_range`: Range of bin counts to explore
- `epsilon_range`: (min, max) epsilon values to explore
- `n_epsilon_samples`: Number of epsilon values to sample
- `time_limit_per_solve`: Time limit for each MILP solve
- `silent`: Whether to suppress output

# Returns
Vector of ParetoPoint representing the Pareto front
"""
function find_pareto_front(
        vp_values::AbstractVector,
        target_data::AbstractVector;
        nbins_range::UnitRange{Int} = 10:5:50,
        epsilon_range::Tuple{Float64, Float64} = (0.01, 0.5),
        n_epsilon_samples::Int = 10,
        time_limit_per_solve::Float64 = 5.0,
        silent::Bool = true
    )
    epsilon_values = range(epsilon_range[1], epsilon_range[2], length = n_epsilon_samples)
    nbins_values = collect(nbins_range)

    all_points = ParetoPoint[]

    for nbins in nbins_values
        # Create target distribution with this nbins
        target = create_target_from_data(target_data; nbins = nbins)

        for eps in epsilon_values
            result = solve_milp_calibration(
                vp_values, target, eps;
                time_limit = time_limit_per_solve,
                silent = silent
            )

            if result.n_selected > 0 && isfinite(result.mean_error)
                push!(
                    all_points, ParetoPoint(
                        nbins, eps, result.n_selected, result.mean_error, result
                    )
                )
            end
        end
    end

    # Filter to Pareto-optimal points
    # A point is dominated if another has more VPs AND lower error
    pareto_points = ParetoPoint[]

    for p in all_points
        is_dominated = false
        for q in all_points
            if q.n_selected > p.n_selected && q.mean_error < p.mean_error
                is_dominated = true
                break
            end
        end
        if !is_dominated
            push!(pareto_points, p)
        end
    end

    # Sort by n_selected (descending)
    sort!(pareto_points, by = p -> -p.n_selected)

    return pareto_points
end

"""
    select_optimal_pareto_point(
        pareto_front::Vector{ParetoPoint};
        method::Symbol = :knee
    ) -> ParetoPoint

Select the optimal point from the Pareto front.

# Methods
- `:knee`: Select the knee point (closest to utopia point in normalized space)
- `:min_error`: Select the point with minimum mean error
- `:max_vps`: Select the point with maximum VPs
"""
function select_optimal_pareto_point(
        pareto_front::Vector{ParetoPoint};
        method::Symbol = :knee
    )
    if isempty(pareto_front)
        error("Empty Pareto front")
    end

    if method == :min_error
        _, idx = findmin(p.mean_error for p in pareto_front)
        return pareto_front[idx]
    elseif method == :max_vps
        _, idx = findmax(p.n_selected for p in pareto_front)
        return pareto_front[idx]
    else  # :knee (default)
        # Normalize objectives and find closest to utopia (1, 0)
        vps = [p.n_selected for p in pareto_front]
        errors = [p.mean_error for p in pareto_front]

        vps_min, vps_max = extrema(vps)
        err_min, err_max = extrema(errors)

        if vps_max == vps_min
            vps_norm = ones(length(vps))
        else
            vps_norm = (vps .- vps_min) ./ (vps_max - vps_min)
        end

        if err_max == err_min
            err_norm = zeros(length(errors))
        else
            err_norm = (errors .- err_min) ./ (err_max - err_min)
        end

        # Distance to utopia point (1, 0)
        distances = sqrt.((1 .- vps_norm) .^ 2 .+ err_norm .^ 2)
        _, idx = findmin(distances)

        return pareto_front[idx]
    end
end

#=============================================================================
# Convenience Functions
=============================================================================#

"""
    calibrate_vpop(
        vpop::DataFrame,
        variable::Symbol,
        target_data::AbstractVector;
        nbins::Int = 20,
        epsilon::Float64 = 0.1,
        time_limit::Float64 = 60.0
    ) -> Tuple{DataFrame, CalibrationResult}

Convenience function to calibrate a virtual population.

# Arguments
- `vpop`: Virtual population DataFrame
- `variable`: Column name to calibrate
- `target_data`: Clinical data to match
- `nbins`: Number of bins for distribution matching
- `epsilon`: Tolerance (0.1 = 10%)
- `time_limit`: Solver time limit

# Returns
Tuple of (calibrated_vpop, calibration_result)
"""
function calibrate_vpop(
        vpop::DataFrame,
        variable::Symbol,
        target_data::AbstractVector;
        nbins::Int = 20,
        epsilon::Float64 = 0.1,
        time_limit::Float64 = 60.0
    )
    vp_values = vpop[!, variable]
    target = create_target_from_data(target_data; nbins = nbins, variable_name = variable)

    result = solve_milp_calibration(vp_values, target, epsilon; time_limit = time_limit)

    if result.n_selected > 0
        calibrated_vpop = vpop[result.selected_indices, :]
        calibrated_vpop.id = 1:nrow(calibrated_vpop)
    else
        calibrated_vpop = DataFrame()
    end

    return calibrated_vpop, result
end

"""
    print_calibration_summary(result::CalibrationResult)

Print a summary of calibration results.
"""
function print_calibration_summary(result::CalibrationResult)
    println("="^50)
    println("MILP Calibration Results")
    println("="^50)
    println("Solver status: $(result.solver_status)")
    println("Original VPs: $(result.n_original)")
    println("Selected VPs: $(result.n_selected)")
    println("Selection rate: $(round(100 * result.selection_rate, digits = 1))%")
    println("Mean error: $(round(result.mean_error, digits = 2))%")

    if !isempty(result.bin_errors)
        println("\nPer-bin errors (%):")
        for (i, err) in enumerate(result.bin_errors)
            println("  Bin $i: $(round(err, digits = 2))%")
        end
    end
    return println("="^50)
end

export TargetDistribution, CalibrationResult, ParetoPoint
export create_target_from_data, create_target_from_specification
export classify_vps_to_bins, create_bin_membership_matrix
export solve_milp_calibration, solve_multivariable_calibration
export find_pareto_front, select_optimal_pareto_point
export calibrate_vpop, print_calibration_summary
