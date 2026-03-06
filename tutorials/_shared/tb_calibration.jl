# Tumor Burden MILP Calibration
# Helper script for calibrating tumor burden virtual populations
# Used by TB-06 onwards

using JuMP
using HiGHS
using DataFrames
using Statistics

# Include vpop generation if not already loaded
if !@isdefined(generate_tb_vpop)
    include("tb_vpop_generation.jl")
end

"""
    simulate_tb_response(vpop; treatment_duration=168.0, N0=1.0)

Simulate tumor burden response for each virtual patient.

Returns DataFrame with added columns: final_tumor, response_category
"""
function simulate_tb_response(vpop::DataFrame; treatment_duration::Float64 = 168.0, N0::Float64 = 1.0)
    results = copy(vpop)

    # Simulate final tumor size for each VP
    final_tumors = Float64[]
    for row in eachrow(vpop)
        f, g, k = row.f, row.g, row.k
        N_final = N0 * f * exp(-k * treatment_duration) + N0 * (1 - f) * exp(g * treatment_duration)
        push!(final_tumors, N_final)
    end

    results[!, :final_tumor] = final_tumors

    # Classify response
    results[!, :response_category] = classify_responses(final_tumors)

    return results
end

"""
    classify_responses(final_tumors; thresholds=(0.14, 0.7, 1.2))

Classify tumor responses into categories.

Response categories:
- CR (Complete Response): tumor < 0.14
- PR (Partial Response): 0.14 ≤ tumor < 0.7
- SD (Stable Disease): 0.7 ≤ tumor < 1.2
- PD (Progressive Disease): tumor ≥ 1.2
"""
function classify_responses(
        final_tumors::Vector{Float64};
        thresholds::Tuple{Float64, Float64, Float64} = (0.14, 0.7, 1.2)
    )
    cr_th, pr_th, sd_th = thresholds

    categories = String[]
    for tumor in final_tumors
        if tumor < cr_th
            push!(categories, "CR")
        elseif tumor < pr_th
            push!(categories, "PR")
        elseif tumor < sd_th
            push!(categories, "SD")
        else
            push!(categories, "PD")
        end
    end

    return categories
end

"""
    calibrate_tb_vpop(vpop, target_pcts; epsilon=0.1)

Calibrate tumor burden vpop to match target response rate distribution using MILP.

# Arguments
- `vpop::DataFrame`: Virtual population with final_tumor column
- `target_pcts::Vector{Float64}`: Target percentages for [CR, PR, SD, PD]
- `epsilon::Float64`: Tolerance for matching (default 0.1 = ±10%)

# Returns
- Calibrated DataFrame with selected virtual patients
"""
function calibrate_tb_vpop(vpop::DataFrame, target_pcts::Vector{Float64}; epsilon::Float64 = 0.1)
    # Ensure response categories exist
    if !hasproperty(vpop, :response_category)
        error("vpop must have response_category column. Run simulate_tb_response first.")
    end

    n_vps = nrow(vpop)
    categories = ["CR", "PR", "SD", "PD"]
    n_cats = length(categories)

    # Create membership matrix
    membership = zeros(Float64, n_cats, n_vps)
    for (j, cat) in enumerate(vpop.response_category)
        for (i, c) in enumerate(categories)
            if cat == c
                membership[i, j] = 1.0
            end
        end
    end

    # Build MILP model
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)

    @variable(model, n_total >= 1, Int)
    @variable(model, x[1:n_vps], Bin)

    @objective(model, Max, n_total)
    @constraint(model, n_total == sum(x))

    # Distribution constraints
    target_fractions = target_pcts ./ 100.0
    for i in 1:n_cats
        p = target_fractions[i]
        if p > 0
            cat_count = sum(membership[i, j] * x[j] for j in 1:n_vps)
            @constraint(model, cat_count <= (1 + epsilon) * p * n_total)
            @constraint(model, cat_count >= (1 - epsilon) * p * n_total)
        end
    end

    @constraint(model, n_total >= n_cats)
    @constraint(model, n_total <= n_vps)

    # Solve
    optimize!(model)
    status = termination_status(model)

    if status in [MOI.OPTIMAL, MOI.FEASIBLE_POINT] ||
            (status == MOI.TIME_LIMIT && has_values(model))
        x_vals = value.(x)
        selected = findall(x_vals .> 0.5)
        calibrated = vpop[selected, :]
        calibrated.id = 1:nrow(calibrated)
        return calibrated
    else
        error("MILP calibration failed: $status")
    end
end

"""
    validate_tb_calibration(calibrated_vpop, target_pcts)

Validate calibration by comparing calibrated vs target distributions.

Returns DataFrame with comparison statistics.
"""
function validate_tb_calibration(calibrated_vpop::DataFrame, target_pcts::Vector{Float64})
    categories = ["CR", "PR", "SD", "PD"]
    n_total = nrow(calibrated_vpop)

    actual_counts = [count(==(c), calibrated_vpop.response_category) for c in categories]
    actual_pcts = 100.0 .* actual_counts ./ n_total

    return DataFrame(
        category = categories,
        target_pct = target_pcts,
        actual_pct = round.(actual_pcts, digits = 1),
        error = round.(abs.(actual_pcts .- target_pcts), digits = 2)
    )
end

# Export functions
export simulate_tb_response, classify_responses, calibrate_tb_vpop, validate_tb_calibration
