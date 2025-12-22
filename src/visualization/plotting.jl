#=
Visualization Module for ISCT Workflow

Provides comprehensive plotting utilities for:
1. Virtual population distributions
2. VCT simulation results
3. MILP calibration results
4. Global sensitivity analysis

Uses AlgebraOfGraphics.jl for declarative, grammar-of-graphics style plotting.

Reference:
    Supporting visualization for the 7-step ISCT workflow
=#

using AlgebraOfGraphics
using CairoMakie
using DataFrames
using DataFramesMeta
using Statistics

#=============================================================================
# Theme and Style Configuration
=============================================================================#

"""
    ISCT_THEME

Default theme for ISCT workflow plots.
"""
const ISCT_THEME = Theme(
    fontsize = 12,
    Axis = (
        titlesize = 14,
        xlabelsize = 12,
        ylabelsize = 12,
        xticklabelsize = 10,
        yticklabelsize = 10
    ),
    Legend = (
        framevisible = false,
        labelsize = 10
    )
)

"""
    TREATMENT_COLORS

Color palette for treatment arms.
"""
const TREATMENT_COLORS = Dict(
    :CONTROL => colorant"#808080",      # Gray
    :NUC_ONLY => colorant"#1f77b4",     # Blue
    :IFN_ONLY => colorant"#ff7f0e",     # Orange
    :NUC_IFN_COMBO => colorant"#2ca02c" # Green
)

"""
    set_isct_theme!()

Apply the ISCT theme globally.
"""
function set_isct_theme!()
    set_theme!(ISCT_THEME)
end

#=============================================================================
# Virtual Population Distribution Plots
=============================================================================#

"""
    plot_parameter_distributions(
        vpop::DataFrame,
        params::Vector{Symbol};
        nbins::Int = 30,
        title::String = "Parameter Distributions"
    ) -> Figure

Create histogram panel for virtual population parameters.

# Arguments
- `vpop`: Virtual population DataFrame
- `params`: Parameter columns to plot
- `nbins`: Number of histogram bins
- `title`: Plot title

# Returns
Makie Figure with parameter histograms
"""
function plot_parameter_distributions(
    vpop::DataFrame,
    params::Vector{Symbol};
    nbins::Int = 30,
    title::String = "Parameter Distributions"
)
    n_params = length(params)
    ncols = min(n_params, 3)
    nrows = ceil(Int, n_params / ncols)

    fig = Figure(size = (300 * ncols, 250 * nrows))

    for (i, param) in enumerate(params)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1

        ax = Axis(fig[row, col],
            xlabel = string(param),
            ylabel = "Count",
            title = string(param)
        )

        hist!(ax, vpop[!, param], bins = nbins, color = (:blue, 0.6))
    end

    Label(fig[0, :], title, fontsize = 16)

    return fig
end

"""
    plot_parameter_correlations(
        vpop::DataFrame,
        params::Vector{Symbol};
        title::String = "Parameter Correlations"
    ) -> Figure

Create scatter plot matrix showing parameter correlations.

# Arguments
- `vpop`: Virtual population DataFrame
- `params`: Parameter columns to plot
- `title`: Plot title

# Returns
Makie Figure with correlation scatter plots
"""
function plot_parameter_correlations(
    vpop::DataFrame,
    params::Vector{Symbol};
    title::String = "Parameter Correlations"
)
    n_params = length(params)
    fig = Figure(size = (200 * n_params, 200 * n_params))

    # Sample for performance if large dataset
    sample_size = min(nrow(vpop), 1000)
    sample_idx = rand(1:nrow(vpop), sample_size)
    vpop_sample = vpop[sample_idx, :]

    for i in 1:n_params
        for j in 1:n_params
            ax = Axis(fig[i, j])

            if i == j
                # Diagonal: histogram
                hist!(ax, vpop_sample[!, params[i]], bins = 20, color = (:blue, 0.6))
            elseif i > j
                # Lower triangle: scatter plot
                scatter!(ax, vpop_sample[!, params[j]], vpop_sample[!, params[i]],
                        markersize = 3, color = (:blue, 0.3))

                # Add correlation coefficient
                r = cor(vpop[!, params[j]], vpop[!, params[i]])
                text!(ax, 0.05, 0.95, text = "r=$(round(r, digits=2))",
                      space = :relative, align = (:left, :top), fontsize = 10)
            else
                # Upper triangle: empty or density
                hidedecorations!(ax)
                hidespines!(ax)
            end

            # Labels
            if i == n_params
                ax.xlabel = string(params[j])
            end
            if j == 1
                ax.ylabel = string(params[i])
            end
        end
    end

    Label(fig[0, :], title, fontsize = 16)

    return fig
end

"""
    plot_vpop_comparison(
        vpop_original::DataFrame,
        vpop_calibrated::DataFrame,
        param::Symbol;
        title::String = "Before vs After Calibration"
    ) -> Figure

Compare parameter distribution before and after calibration.
"""
function plot_vpop_comparison(
    vpop_original::DataFrame,
    vpop_calibrated::DataFrame,
    param::Symbol;
    nbins::Int = 30,
    title::String = "Before vs After Calibration"
)
    fig = Figure(size = (700, 400))

    ax1 = Axis(fig[1, 1],
        xlabel = string(param),
        ylabel = "Density",
        title = "Original (n=$(nrow(vpop_original)))"
    )

    ax2 = Axis(fig[1, 2],
        xlabel = string(param),
        ylabel = "Density",
        title = "Calibrated (n=$(nrow(vpop_calibrated)))"
    )

    hist!(ax1, vpop_original[!, param], bins = nbins,
          normalization = :pdf, color = (:blue, 0.6))
    hist!(ax2, vpop_calibrated[!, param], bins = nbins,
          normalization = :pdf, color = (:green, 0.6))

    # Link axes
    linkyaxes!(ax1, ax2)

    Label(fig[0, :], title, fontsize = 16)

    return fig
end

#=============================================================================
# VCT Simulation Result Plots
=============================================================================#

"""
    plot_tumor_dynamics(
        treatment_df::DataFrame,
        control_df::DataFrame;
        time_col::Symbol = :time,
        value_col::Symbol = :Nt,
        title::String = "Tumor Burden Dynamics"
    ) -> Figure

Plot tumor burden dynamics for treatment vs control arms.

# Arguments
- `treatment_df`: Treatment arm simulation results
- `control_df`: Control arm simulation results
- `time_col`: Column name for time
- `value_col`: Column name for tumor size
- `title`: Plot title

# Returns
Makie Figure with median and IQR bands
"""
function plot_tumor_dynamics(
    treatment_df::DataFrame,
    control_df::DataFrame;
    time_col::Symbol = :time,
    value_col::Symbol = :Nt,
    title::String = "Tumor Burden Dynamics"
)
    # Calculate summary statistics
    function summarize_by_time(df, time_col, value_col)
        @chain df begin
            @groupby(cols(time_col))
            @combine(
                :median = median(cols(value_col)),
                :q25 = quantile(cols(value_col), 0.25),
                :q75 = quantile(cols(value_col), 0.75),
                :q05 = quantile(cols(value_col), 0.05),
                :q95 = quantile(cols(value_col), 0.95)
            )
            @transform(:time_weeks = cols(time_col) ./ 7)
        end
    end

    tx_summary = summarize_by_time(treatment_df, time_col, value_col)
    ctrl_summary = summarize_by_time(control_df, time_col, value_col)

    fig = Figure(size = (700, 500))

    ax = Axis(fig[1, 1],
        xlabel = "Time (weeks)",
        ylabel = "Baseline-Normalized Tumor Diameter",
        title = title
    )

    # Control arm (90% CI band)
    band!(ax, ctrl_summary.time_weeks, ctrl_summary.q05, ctrl_summary.q95,
          color = (:gray, 0.2))
    # Control arm (IQR band)
    band!(ax, ctrl_summary.time_weeks, ctrl_summary.q25, ctrl_summary.q75,
          color = (:gray, 0.3))
    # Control median
    lines!(ax, ctrl_summary.time_weeks, ctrl_summary.median,
           color = :gray, linewidth = 2, label = "Control")

    # Treatment arm (90% CI band)
    band!(ax, tx_summary.time_weeks, tx_summary.q05, tx_summary.q95,
          color = (:blue, 0.2))
    # Treatment arm (IQR band)
    band!(ax, tx_summary.time_weeks, tx_summary.q25, tx_summary.q75,
          color = (:blue, 0.3))
    # Treatment median
    lines!(ax, tx_summary.time_weeks, tx_summary.median,
           color = :blue, linewidth = 2, label = "Treatment")

    axislegend(ax, position = :lt)

    return fig
end

"""
    plot_response_waterfall(
        sim_results::DataFrame,
        baseline_time::Real,
        final_time::Real;
        id_col::Symbol = :id,
        time_col::Symbol = :time,
        value_col::Symbol = :Nt,
        threshold::Float64 = 0.14,
        title::String = "Response Waterfall Plot"
    ) -> Figure

Create waterfall plot showing percent change from baseline.
"""
function plot_response_waterfall(
    sim_results::DataFrame,
    baseline_time::Real,
    final_time::Real;
    id_col::Symbol = :id,
    time_col::Symbol = :time,
    value_col::Symbol = :Nt,
    threshold::Float64 = 0.14,
    title::String = "Response Waterfall Plot",
    max_patients::Int = 100
)
    # Calculate percent change
    baseline = @chain sim_results begin
        @subset(cols(time_col) .== baseline_time)
        @select(cols(id_col), baseline = cols(value_col))
    end

    final = @chain sim_results begin
        @subset(cols(time_col) .== final_time)
        @select(cols(id_col), final = cols(value_col))
    end

    changes = innerjoin(baseline, final, on = id_col)
    changes[!, :pct_change] = (changes.final .- changes.baseline) ./ changes.baseline .* 100

    # Sort by percent change
    sort!(changes, :pct_change)

    # Limit number of patients for visualization
    if nrow(changes) > max_patients
        sample_idx = round.(Int, range(1, nrow(changes), length=max_patients))
        changes = changes[sample_idx, :]
    end

    fig = Figure(size = (800, 500))

    ax = Axis(fig[1, 1],
        xlabel = "Patient",
        ylabel = "Change from Baseline (%)",
        title = title
    )

    # Color by response
    colors = [c < (threshold - 1) * 100 ? :green : (c > 20 ? :red : :gray)
              for c in changes.pct_change]

    barplot!(ax, 1:nrow(changes), changes.pct_change, color = colors)

    # Add reference lines
    hlines!(ax, [0], color = :black, linestyle = :solid)
    hlines!(ax, [-30], color = :blue, linestyle = :dash, label = "PR (-30%)")
    hlines!(ax, [20], color = :red, linestyle = :dash, label = "PD (+20%)")

    return fig
end

"""
    plot_treatment_comparison(
        results_dict::Dict,
        endpoint::Symbol;
        title::String = "Treatment Comparison"
    ) -> Figure

Create bar chart comparing endpoints across treatment arms.

# Arguments
- `results_dict`: Dict mapping treatment name to (rate, ci_lower, ci_upper)
- `endpoint`: Name of endpoint being compared
- `title`: Plot title
"""
function plot_treatment_comparison(
    results_dict::Dict,
    endpoint::Symbol;
    title::String = "Treatment Comparison"
)
    treatments = collect(keys(results_dict))
    n_treatments = length(treatments)

    rates = [results_dict[t][1] for t in treatments]
    ci_low = [results_dict[t][2] for t in treatments]
    ci_high = [results_dict[t][3] for t in treatments]

    fig = Figure(size = (600, 450))

    ax = Axis(fig[1, 1],
        xlabel = "Treatment Arm",
        ylabel = "$(endpoint) Rate (%)",
        title = title,
        xticks = (1:n_treatments, string.(treatments))
    )

    colors = [get(TREATMENT_COLORS, Symbol(t), colorant"#999999") for t in treatments]

    barplot!(ax, 1:n_treatments, rates, color = colors)
    errorbars!(ax, 1:n_treatments, rates, rates .- ci_low, ci_high .- rates,
               color = :black, whiskerwidth = 10)

    return fig
end

"""
    plot_hbv_dynamics(
        dynamics_df::DataFrame;
        id_sample::Int = 10,
        title::String = "HBV Viral Dynamics"
    ) -> Figure

Plot HBV viral and HBsAg dynamics for sample of patients.
"""
function plot_hbv_dynamics(
    dynamics_df::DataFrame;
    id_sample::Int = 10,
    title::String = "HBV Viral Dynamics"
)
    # Sample patients
    unique_ids = unique(dynamics_df.id)
    sample_ids = unique_ids[1:min(id_sample, length(unique_ids))]

    sample_df = @subset(dynamics_df, :id .∈ Ref(sample_ids))

    fig = Figure(size = (800, 600))

    # HBsAg panel
    ax1 = Axis(fig[1, 1],
        xlabel = "Time (days)",
        ylabel = "log₁₀(HBsAg) IU/mL",
        title = "HBsAg Dynamics"
    )

    if :log_HBsAg in names(dynamics_df)
        for id in sample_ids
            patient_data = @subset(sample_df, :id .== id)
            lines!(ax1, patient_data.time, patient_data.log_HBsAg,
                   color = (:blue, 0.3), linewidth = 0.5)
        end
        hlines!(ax1, [log10(0.05)], color = :red, linestyle = :dash,
                label = "LOQ (0.05 IU/mL)")
    end

    # Viral load panel
    ax2 = Axis(fig[2, 1],
        xlabel = "Time (days)",
        ylabel = "log₁₀(HBV DNA) copies/mL",
        title = "Viral Load Dynamics"
    )

    if :log_V in names(dynamics_df)
        for id in sample_ids
            patient_data = @subset(sample_df, :id .== id)
            lines!(ax2, patient_data.time, patient_data.log_V,
                   color = (:orange, 0.3), linewidth = 0.5)
        end
        hlines!(ax2, [log10(25)], color = :red, linestyle = :dash,
                label = "LOQ (25 copies/mL)")
    end

    Label(fig[0, :], title, fontsize = 16)

    return fig
end

#=============================================================================
# MILP Calibration Plots
=============================================================================#

"""
    plot_calibration_result(
        original_values::AbstractVector,
        calibrated_values::AbstractVector,
        target::TargetDistribution;
        title::String = "MILP Calibration Result"
    ) -> Figure

Compare original, calibrated, and target distributions.
"""
function plot_calibration_result(
    original_values::AbstractVector,
    calibrated_values::AbstractVector,
    target;
    title::String = "MILP Calibration Result"
)
    fig = Figure(size = (800, 600))

    # Original distribution
    ax1 = Axis(fig[1, 1],
        xlabel = string(target.variable_name),
        ylabel = "Percentage (%)",
        title = "Original (n=$(length(original_values)))"
    )

    hist!(ax1, original_values, bins = length(target.percentages),
          normalization = :probability, color = (:blue, 0.5))

    # Calibrated distribution
    ax2 = Axis(fig[2, 1],
        xlabel = string(target.variable_name),
        ylabel = "Percentage (%)",
        title = "Calibrated (n=$(length(calibrated_values)))"
    )

    hist!(ax2, calibrated_values, bins = length(target.percentages),
          normalization = :probability, color = (:green, 0.5))

    # Target distribution (bar chart)
    ax3 = Axis(fig[3, 1],
        xlabel = string(target.variable_name),
        ylabel = "Percentage (%)",
        title = "Target Distribution"
    )

    bin_centers = [(target.edges[i] + target.edges[i+1])/2 for i in 1:length(target.percentages)]
    barplot!(ax3, 1:length(target.percentages), target.percentages,
             color = (:orange, 0.7))

    Label(fig[0, :], title, fontsize = 16)

    return fig
end

"""
    plot_pareto_front(
        pareto_points::Vector;
        optimal_idx::Union{Int, Nothing} = nothing,
        title::String = "Pareto Front"
    ) -> Figure

Plot Pareto front for MILP calibration.
"""
function plot_pareto_front(
    pareto_points::Vector;
    optimal_idx::Union{Int, Nothing} = nothing,
    title::String = "Pareto Front: VPs vs Distribution Error"
)
    fig = Figure(size = (600, 450))

    ax = Axis(fig[1, 1],
        xlabel = "Number of Selected VPs",
        ylabel = "Mean Distribution Error (%)",
        title = title
    )

    vps = [p.n_selected for p in pareto_points]
    errors = [p.mean_error for p in pareto_points]

    scatter!(ax, vps, errors, color = :blue, markersize = 8, label = "Pareto points")
    lines!(ax, vps, errors, color = (:blue, 0.3))

    if !isnothing(optimal_idx) && optimal_idx <= length(pareto_points)
        scatter!(ax, [vps[optimal_idx]], [errors[optimal_idx]],
                 color = :red, markersize = 15, marker = :star5, label = "Optimal")
    end

    axislegend(ax, position = :rt)

    return fig
end

#=============================================================================
# GSA Visualization Plots
=============================================================================#

"""
    plot_gsa_indices(
        gsa_result::GSAResult,
        output::Symbol;
        title::String = "Sensitivity Indices"
    ) -> Figure

Create bar chart of GSA sensitivity indices.
"""
function plot_gsa_indices(
    gsa_result,
    output::Symbol;
    title::String = "Sensitivity Indices"
)
    # Extract data for this output
    fo = filter(row -> row.output == output, gsa_result.first_order)
    to = filter(row -> row.output == output, gsa_result.total_order)

    # Sort by total order
    sort!(to, :index, rev = true)
    param_order = to.parameter

    # Reorder first order to match
    fo_ordered = [filter(r -> r.parameter == p, fo)[1, :index] for p in param_order]
    to_ordered = [filter(r -> r.parameter == p, to)[1, :index] for p in param_order]

    n_params = length(param_order)

    fig = Figure(size = (700, 450))

    ax = Axis(fig[1, 1],
        xlabel = "Parameter",
        ylabel = "Sensitivity Index",
        title = "$title: $output",
        xticks = (1:n_params, string.(param_order)),
        xticklabelrotation = π/6
    )

    barwidth = 0.35
    barplot!(ax, (1:n_params) .- barwidth/2, fo_ordered,
             width = barwidth, color = :blue, label = "First Order (S₁)")
    barplot!(ax, (1:n_params) .+ barwidth/2, to_ordered,
             width = barwidth, color = :orange, label = "Total Order (Sₜ)")

    hlines!(ax, [0.1], color = :red, linestyle = :dash, label = "Threshold")

    axislegend(ax, position = :rt)

    return fig
end

"""
    plot_gsa_heatmap(
        gsa_result::GSAResult;
        index_type::Symbol = :total_order,
        title::String = "GSA Heatmap"
    ) -> Figure

Create heatmap of sensitivity indices across outputs.
"""
function plot_gsa_heatmap(
    gsa_result;
    index_type::Symbol = :total_order,
    title::String = "GSA Heatmap"
)
    indices_df = index_type == :total_order ? gsa_result.total_order : gsa_result.first_order

    outputs = gsa_result.outputs
    params = gsa_result.parameters

    # Build matrix
    matrix = zeros(length(outputs), length(params))
    for (i, output) in enumerate(outputs)
        for (j, param) in enumerate(params)
            row = filter(r -> r.output == output && r.parameter == param, indices_df)
            if nrow(row) > 0
                matrix[i, j] = row[1, :index]
            end
        end
    end

    fig = Figure(size = (600, 400))

    ax = Axis(fig[1, 1],
        xlabel = "Parameter",
        ylabel = "Output",
        title = title,
        xticks = (1:length(params), string.(params)),
        yticks = (1:length(outputs), string.(outputs)),
        xticklabelrotation = π/4
    )

    hm = heatmap!(ax, 1:length(params), 1:length(outputs), matrix',
                  colormap = :viridis)

    Colorbar(fig[1, 2], hm, label = "Sensitivity Index")

    return fig
end

"""
    plot_gsa_comparison(
        gsa_results::Dict{Symbol, Any},
        output::Symbol;
        title::String = "GSA Treatment Comparison"
    ) -> Figure

Compare GSA results across different treatments/conditions.
"""
function plot_gsa_comparison(
    gsa_results::Dict,
    output::Symbol;
    title::String = "GSA Treatment Comparison"
)
    treatments = collect(keys(gsa_results))
    n_treatments = length(treatments)

    # Get parameter names from first result
    first_result = gsa_results[first(treatments)]
    params = first_result.parameters
    n_params = length(params)

    # Extract total order indices
    function get_indices(result, output)
        to = filter(row -> row.output == output, result.total_order)
        return [filter(r -> r.parameter == p, to)[1, :index] for p in params]
    end

    fig = Figure(size = (800, 500))

    ax = Axis(fig[1, 1],
        xlabel = "Parameter",
        ylabel = "Total Order Index",
        title = title,
        xticks = (1:n_params, string.(params)),
        xticklabelrotation = π/4
    )

    barwidth = 0.8 / n_treatments
    offsets = range(-0.4 + barwidth/2, 0.4 - barwidth/2, length = n_treatments)
    colors = [:blue, :orange, :green, :purple, :red]

    for (i, treatment) in enumerate(treatments)
        indices = get_indices(gsa_results[treatment], output)
        barplot!(ax, (1:n_params) .+ offsets[i], indices,
                 width = barwidth, color = colors[mod1(i, length(colors))],
                 label = string(treatment))
    end

    axislegend(ax, position = :rt)

    return fig
end

#=============================================================================
# Composite/Summary Plots
=============================================================================#

"""
    create_isct_summary_figure(
        vpop::DataFrame,
        vct_results,
        gsa_result;
        title::String = "ISCT Workflow Summary"
    ) -> Figure

Create comprehensive summary figure for ISCT results.
"""
function create_isct_summary_figure(
    vpop::DataFrame,
    params::Vector{Symbol},
    treatment_df::DataFrame,
    control_df::DataFrame;
    title::String = "ISCT Workflow Summary"
)
    fig = Figure(size = (1200, 800))

    # Panel A: Parameter distributions
    n_params = min(length(params), 3)
    for (i, param) in enumerate(params[1:n_params])
        ax = Axis(fig[1, i],
            xlabel = string(param),
            ylabel = "Count",
            title = "$(param) Distribution"
        )
        hist!(ax, vpop[!, param], bins = 20, color = (:blue, 0.6))
    end

    # Panel B: VCT dynamics
    ax_vct = Axis(fig[2, 1:2],
        xlabel = "Time (weeks)",
        ylabel = "Tumor Size",
        title = "VCT Dynamics"
    )

    # Summarize by time
    tx_summary = @chain treatment_df begin
        @groupby(:time)
        @combine(:median = median(:Nt), :q25 = quantile(:Nt, 0.25), :q75 = quantile(:Nt, 0.75))
        @transform(:weeks = :time ./ 7)
    end

    ctrl_summary = @chain control_df begin
        @groupby(:time)
        @combine(:median = median(:Nt), :q25 = quantile(:Nt, 0.25), :q75 = quantile(:Nt, 0.75))
        @transform(:weeks = :time ./ 7)
    end

    band!(ax_vct, ctrl_summary.weeks, ctrl_summary.q25, ctrl_summary.q75, color = (:gray, 0.3))
    lines!(ax_vct, ctrl_summary.weeks, ctrl_summary.median, color = :gray, linewidth = 2, label = "Control")

    band!(ax_vct, tx_summary.weeks, tx_summary.q25, tx_summary.q75, color = (:blue, 0.3))
    lines!(ax_vct, tx_summary.weeks, tx_summary.median, color = :blue, linewidth = 2, label = "Treatment")

    axislegend(ax_vct, position = :lt)

    # Panel C: Response summary (placeholder)
    ax_response = Axis(fig[2, 3],
        xlabel = "Arm",
        ylabel = "Response Rate (%)",
        title = "Response at Week 18",
        xticks = (1:2, ["Control", "Treatment"])
    )

    # Calculate response rates
    final_tx = @subset(treatment_df, :time .== maximum(:time))
    final_ctrl = @subset(control_df, :time .== maximum(:time))

    tx_rate = 100 * sum(final_tx.Nt .< 0.14) / nrow(final_tx)
    ctrl_rate = 100 * sum(final_ctrl.Nt .< 0.14) / nrow(final_ctrl)

    barplot!(ax_response, [1, 2], [ctrl_rate, tx_rate], color = [:gray, :blue])

    Label(fig[0, :], title, fontsize = 18)

    return fig
end

#=============================================================================
# Utility Functions
=============================================================================#

"""
    save_figure(fig::Figure, filepath::String; px_per_unit::Int = 2)

Save figure to file with consistent settings.
"""
function save_figure(fig::Figure, filepath::String; px_per_unit::Int = 2)
    save(filepath, fig, px_per_unit = px_per_unit)
    println("Saved: $filepath")
end

"""
    quick_hist(values::AbstractVector, title::String = "Distribution") -> Figure

Create quick histogram for exploration.
"""
function quick_hist(values::AbstractVector; title::String = "Distribution", bins::Int = 30)
    fig = Figure(size = (500, 400))
    ax = Axis(fig[1, 1], xlabel = "Value", ylabel = "Count", title = title)
    hist!(ax, values, bins = bins, color = (:blue, 0.6))
    return fig
end

"""
    quick_scatter(x::AbstractVector, y::AbstractVector;
                  xlabel::String = "x", ylabel::String = "y") -> Figure

Create quick scatter plot for exploration.
"""
function quick_scatter(x::AbstractVector, y::AbstractVector;
                       xlabel::String = "x", ylabel::String = "y",
                       title::String = "Scatter Plot")
    fig = Figure(size = (500, 400))
    ax = Axis(fig[1, 1], xlabel = xlabel, ylabel = ylabel, title = title)
    scatter!(ax, x, y, markersize = 5, color = (:blue, 0.5))
    return fig
end

#=============================================================================
# Exports
=============================================================================#

export ISCT_THEME, TREATMENT_COLORS, set_isct_theme!

# Vpop distribution plots
export plot_parameter_distributions, plot_parameter_correlations, plot_vpop_comparison

# VCT result plots
export plot_tumor_dynamics, plot_response_waterfall, plot_treatment_comparison
export plot_hbv_dynamics

# Calibration plots
export plot_calibration_result, plot_pareto_front

# GSA plots
export plot_gsa_indices, plot_gsa_heatmap, plot_gsa_comparison

# Summary and utility
export create_isct_summary_figure, save_figure, quick_hist, quick_scatter
