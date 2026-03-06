# HBV MILP Calibration
# Helper script for calibrating HBV virtual populations to baseline HBsAg
# Used by HBV-06 onwards

using JuMP
using HiGHS
using DataFrames
using Statistics

# Include vpop generation if not already loaded
if !@isdefined(generate_hbv_vpop)
    include("hbv_vpop_generation.jl")
end

# Everest trial baseline HBsAg distribution
const EVEREST_TARGET = (
    edges = [log10(0.05), log10(100), log10(200), log10(500), log10(1000), log10(10000)],
    percentages = [37.5, 10.9, 19.1, 21.5, 11.0],
    bin_labels = ["<100", "100-200", "200-500", "500-1000", ">1000"],
)

"""
    calibrate_hbsag_milp(vpop, target_edges, target_pcts; epsilon=0.1)

Calibrate HBV vpop to match target HBsAg distribution using MILP.

# Arguments
- `vpop::DataFrame`: Virtual population with log_hbsag_bl column
- `target_edges::Vector{Float64}`: Bin edges (log10 scale)
- `target_pcts::Vector{Float64}`: Target percentages for each bin
- `epsilon::Float64`: Tolerance for matching (default 0.1 = ±10%)

# Returns
- Calibrated DataFrame with selected virtual patients
"""
function calibrate_hbsag_milp(
        vpop::DataFrame,
        target_edges::Vector{Float64},
        target_pcts::Vector{Float64};
        epsilon::Float64 = 0.1
    )
    # Ensure baseline HBsAg exists
    if !hasproperty(vpop, :log_hbsag_bl)
        add_baseline_hbsag!(vpop)
    end

    n_vps = nrow(vpop)
    n_bins = length(target_pcts)
    values = vpop.log_hbsag_bl

    # Assign VPs to bins
    bin_assignments = zeros(Int, n_vps)
    for i in 1:n_vps
        for b in 1:n_bins
            if values[i] >= target_edges[b] && values[i] < target_edges[b + 1]
                bin_assignments[i] = b
                break
            end
        end
    end

    # Create membership matrix
    membership = zeros(Float64, n_bins, n_vps)
    for (j, bin) in enumerate(bin_assignments)
        if bin > 0
            membership[bin, j] = 1.0
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
    for b in 1:n_bins
        p_b = target_fractions[b]
        if p_b > 0
            bin_count = sum(membership[b, j] * x[j] for j in 1:n_vps)
            @constraint(model, bin_count <= (1 + epsilon) * p_b * n_total)
            @constraint(model, bin_count >= (1 - epsilon) * p_b * n_total)
        end
    end

    @constraint(model, n_total >= n_bins)
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
    calibrate_to_everest(vpop; epsilon=0.1)

Convenience function to calibrate HBV vpop to Everest trial distribution.
"""
function calibrate_to_everest(vpop::DataFrame; epsilon::Float64 = 0.1)
    return calibrate_hbsag_milp(
        vpop,
        EVEREST_TARGET.edges,
        EVEREST_TARGET.percentages;
        epsilon = epsilon
    )
end

"""
    validate_hbsag_calibration(calibrated_vpop, target_edges, target_pcts)

Validate HBsAg calibration by comparing calibrated vs target distributions.
"""
function validate_hbsag_calibration(
        calibrated_vpop::DataFrame,
        target_edges::Vector{Float64},
        target_pcts::Vector{Float64}
    )
    n_bins = length(target_pcts)
    n_total = nrow(calibrated_vpop)
    values = calibrated_vpop.log_hbsag_bl

    # Count VPs in each bin
    actual_counts = zeros(Int, n_bins)
    for val in values
        for b in 1:n_bins
            if val >= target_edges[b] && val < target_edges[b + 1]
                actual_counts[b] += 1
                break
            end
        end
    end

    actual_pcts = 100.0 .* actual_counts ./ n_total

    return DataFrame(
        bin = 1:n_bins,
        target_pct = target_pcts,
        actual_pct = round.(actual_pcts, digits = 1),
        error = round.(abs.(actual_pcts .- target_pcts), digits = 2)
    )
end

"""
    validate_everest_calibration(calibrated_vpop)

Convenience function to validate against Everest distribution.
"""
function validate_everest_calibration(calibrated_vpop::DataFrame)
    validation = validate_hbsag_calibration(
        calibrated_vpop,
        EVEREST_TARGET.edges,
        EVEREST_TARGET.percentages
    )
    validation[!, :range] = EVEREST_TARGET.bin_labels
    return validation
end

# Export functions
export calibrate_hbsag_milp, calibrate_to_everest
export validate_hbsag_calibration, validate_everest_calibration
export EVEREST_TARGET
