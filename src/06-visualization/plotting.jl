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
    HBV_OUTCOME_COLORS

Color palette for acute vs chronic infection outcomes.
"""
const HBV_OUTCOME_COLORS = Dict(
    :acute => colorant"#2ca02c",        # Green (cleared)
    :chronic => colorant"#d62728"       # Red (persistent)
)

"""
    HBV_BIOMARKER_LABELS

Display labels for HBV biomarkers.
"""
const HBV_BIOMARKER_LABELS = Dict(
    :log_HBsAg => "log₁₀(HBsAg) IU/mL",
    :log_V => "log₁₀(HBV DNA) copies/mL",
    :log_ALT => "log₁₀(ALT) U/L",
    :log_E => "log₁₀(Effector T cells)"
)

"""
    HBV_LOQ_THRESHOLDS

Limit of quantification thresholds for HBV biomarkers.
"""
const HBV_LOQ_THRESHOLDS = Dict(
    :log_HBsAg => log10(0.05),  # -1.301
    :log_V => log10(25.0)       # 1.398
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
    plot_parameter_distributions_aog(
        vpop::DataFrame,
        params::Vector{Symbol};
        nbins::Int = 30,
        title::String = "Parameter Distributions"
    ) -> Figure

AlgebraOfGraphics version: Create histogram panel for virtual population parameters.

Uses data() |> mapping() |> visual() |> draw() pattern.
"""
function plot_parameter_distributions_aog(
    vpop::DataFrame,
    params::Vector{Symbol};
    nbins::Int = 30,
    title::String = "Parameter Distributions"
)
    # Stack parameters into long format for AOG faceting
    long_df = stack(vpop[:, params], params, variable_name=:parameter, value_name=:value)

    # Create the plot specification
    plt = data(long_df) *
          mapping(:value) *
          histogram(bins=nbins) *
          mapping(col=:parameter)

    # Draw with layout
    fig = draw(
        plt;
        axis = (ylabel = "Count",),
        figure = (title = title,),
        facet = (linkxaxes = :none, linkyaxes = :minimal)
    )

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
    plot_parameter_correlations_aog(
        vpop::DataFrame,
        params::Vector{Symbol};
        title::String = "Parameter Correlations"
    ) -> Figure

AlgebraOfGraphics version: Create pairplot showing parameter correlations.

Uses AOG's pairplot functionality for scatter matrix with histograms on diagonal.
"""
function plot_parameter_correlations_aog(
    vpop::DataFrame,
    params::Vector{Symbol};
    title::String = "Parameter Correlations"
)
    # Sample for performance if large dataset
    sample_size = min(nrow(vpop), 1000)
    sample_idx = rand(1:nrow(vpop), sample_size)
    vpop_sample = vpop[sample_idx, params]

    # Create pairplot specification
    # Diagonal: histograms
    # Off-diagonal: scatter plots
    layers = data(vpop_sample) * mapping(params..., params...)

    # Build the layers: histogram on diagonal, scatter elsewhere
    diag = mapping(params) * histogram(bins=20)
    offdiag = mapping(params..., params...) * visual(Scatter, markersize=3, alpha=0.3)

    # Alternative simpler approach using basic AOG patterns
    # Create pairs plot manually
    n_params = length(params)
    fig = Figure(size = (200 * n_params, 200 * n_params))

    for i in 1:n_params
        for j in 1:n_params
            if i == j
                # Diagonal: histogram using AOG
                plt = data(vpop_sample) * mapping(params[i]) * histogram(bins=20)
                ag = draw!(fig[i, j], plt)
            elseif i > j
                # Lower triangle: scatter plot using AOG
                plt = data(vpop_sample) * mapping(params[j], params[i]) *
                      visual(Scatter, markersize=3, alpha=0.3)
                ag = draw!(fig[i, j], plt)
            else
                # Upper triangle: hide
                ax = Axis(fig[i, j])
                hidedecorations!(ax)
                hidespines!(ax)
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

"""
    plot_vpop_comparison_aog(
        vpop_original::DataFrame,
        vpop_calibrated::DataFrame,
        param::Symbol;
        nbins::Int = 30,
        title::String = "Before vs After Calibration"
    ) -> Figure

AlgebraOfGraphics version: Compare parameter distribution before and after calibration.

Uses AOG faceting with combined DataFrame.
"""
function plot_vpop_comparison_aog(
    vpop_original::DataFrame,
    vpop_calibrated::DataFrame,
    param::Symbol;
    nbins::Int = 30,
    title::String = "Before vs After Calibration"
)
    # Combine data with source label
    df_orig = DataFrame(
        value = vpop_original[!, param],
        source = fill("Original (n=$(nrow(vpop_original)))", nrow(vpop_original))
    )
    df_calib = DataFrame(
        value = vpop_calibrated[!, param],
        source = fill("Calibrated (n=$(nrow(vpop_calibrated)))", nrow(vpop_calibrated))
    )
    combined_df = vcat(df_orig, df_calib)

    # Create AOG plot with faceting
    plt = data(combined_df) *
          mapping(:value) *
          histogram(bins=nbins, normalization=:pdf) *
          mapping(col=:source => sorter(["Original (n=$(nrow(vpop_original)))", "Calibrated (n=$(nrow(vpop_calibrated)))"]))

    fig = draw(
        plt;
        axis = (xlabel = string(param), ylabel = "Density"),
        figure = (title = title,)
    )

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
    plot_tumor_dynamics_aog(
        treatment_df::DataFrame,
        control_df::DataFrame;
        time_col::Symbol = :time,
        value_col::Symbol = :Nt,
        title::String = "Tumor Burden Dynamics"
    ) -> Figure

AlgebraOfGraphics version: Plot tumor burden dynamics for treatment vs control arms.

Uses AOG layering with summary statistics computed beforehand.
"""
function plot_tumor_dynamics_aog(
    treatment_df::DataFrame,
    control_df::DataFrame;
    time_col::Symbol = :time,
    value_col::Symbol = :Nt,
    title::String = "Tumor Burden Dynamics"
)
    # Calculate summary statistics with arm labels
    function summarize_with_arm(df, arm_name)
        @chain df begin
            @groupby(cols(time_col))
            @combine(
                :median = median(cols(value_col)),
                :q25 = quantile(cols(value_col), 0.25),
                :q75 = quantile(cols(value_col), 0.75),
                :q05 = quantile(cols(value_col), 0.05),
                :q95 = quantile(cols(value_col), 0.95)
            )
            @transform(:time_weeks = cols(time_col) ./ 7, :arm = arm_name)
        end
    end

    tx_summary = summarize_with_arm(treatment_df, "Treatment")
    ctrl_summary = summarize_with_arm(control_df, "Control")
    combined = vcat(ctrl_summary, tx_summary)

    # Create the plot layers
    # Median lines with color by arm
    line_layer = data(combined) *
                 mapping(:time_weeks => "Time (weeks)", :median => "Tumor Size",
                        color=:arm => "Arm") *
                 visual(Lines, linewidth=2)

    # Band layer for IQR (Note: AOG doesn't have native band, so we use separate approach)
    # For bands, we'll use a hybrid approach
    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1], xlabel = "Time (weeks)", ylabel = "Baseline-Normalized Tumor Diameter", title = title)

    # Draw bands manually (AOG doesn't directly support ribbon/band)
    for (summary, color) in [(ctrl_summary, :gray), (tx_summary, :blue)]
        band!(ax, summary.time_weeks, summary.q05, summary.q95, color = (color, 0.2))
        band!(ax, summary.time_weeks, summary.q25, summary.q75, color = (color, 0.3))
    end

    # Draw lines using AOG on top
    draw!(fig[1, 1], line_layer)

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
    plot_response_waterfall_aog(
        sim_results::DataFrame,
        baseline_time::Real,
        final_time::Real;
        id_col::Symbol = :id,
        time_col::Symbol = :time,
        value_col::Symbol = :Nt,
        threshold::Float64 = 0.14,
        title::String = "Response Waterfall Plot",
        max_patients::Int = 100
    ) -> Figure

AlgebraOfGraphics version: Create waterfall plot showing percent change from baseline.

Uses AOG with computed response categories for coloring.
"""
function plot_response_waterfall_aog(
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

    # Sort and sample
    sort!(changes, :pct_change)
    if nrow(changes) > max_patients
        sample_idx = round.(Int, range(1, nrow(changes), length=max_patients))
        changes = changes[sample_idx, :]
    end

    # Add response category and patient order
    changes[!, :response] = [c < (threshold - 1) * 100 ? "CR" : (c > 20 ? "PD" : "SD")
                             for c in changes.pct_change]
    changes[!, :patient_order] = 1:nrow(changes)

    # Create AOG bar plot with color by response
    plt = data(changes) *
          mapping(:patient_order => "Patient", :pct_change => "Change from Baseline (%)",
                  color=:response => "Response") *
          visual(BarPlot)

    fig = draw(plt; axis = (title = title,))

    # Add reference lines
    ax = current_axis()
    hlines!(ax, [0], color = :black, linestyle = :solid)
    hlines!(ax, [-30], color = :blue, linestyle = :dash)
    hlines!(ax, [20], color = :red, linestyle = :dash)

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
    plot_treatment_comparison_aog(
        results_dict::Dict,
        endpoint::Symbol;
        title::String = "Treatment Comparison"
    ) -> Figure

AlgebraOfGraphics version: Create bar chart comparing endpoints across treatment arms.

Uses AOG with DataFrame of results for cleaner declarative syntax.
"""
function plot_treatment_comparison_aog(
    results_dict::Dict,
    endpoint::Symbol;
    title::String = "Treatment Comparison"
)
    # Build DataFrame from results
    df = DataFrame(
        treatment = String[],
        rate = Float64[],
        ci_low = Float64[],
        ci_high = Float64[]
    )

    for (t, vals) in results_dict
        push!(df, (string(t), vals[1], vals[2], vals[3]))
    end

    # Create AOG bar plot
    plt = data(df) *
          mapping(:treatment => "Treatment Arm", :rate => "$(endpoint) Rate (%)",
                  color=:treatment => "Treatment") *
          visual(BarPlot)

    fig = draw(plt; axis = (title = title,))

    # Add error bars manually (AOG doesn't directly support asymmetric error bars)
    ax = current_axis()
    x_positions = 1:nrow(df)
    errorbars!(ax, x_positions, df.rate, df.rate .- df.ci_low, df.ci_high .- df.rate,
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

"""
    plot_hbv_dynamics_aog(
        dynamics_df::DataFrame;
        id_sample::Int = 10,
        title::String = "HBV Viral Dynamics"
    ) -> Figure

AlgebraOfGraphics version: Plot HBV viral and HBsAg dynamics for sample of patients.

Uses AOG with group aesthetic for individual patient trajectories.
"""
function plot_hbv_dynamics_aog(
    dynamics_df::DataFrame;
    id_sample::Int = 10,
    title::String = "HBV Viral Dynamics"
)
    # Sample patients
    unique_ids = unique(dynamics_df.id)
    sample_ids = unique_ids[1:min(id_sample, length(unique_ids))]
    sample_df = @subset(dynamics_df, :id .∈ Ref(sample_ids))

    fig = Figure(size = (800, 600))

    # HBsAg panel using AOG
    if :log_HBsAg in names(dynamics_df)
        plt_hbsag = data(sample_df) *
                    mapping(:time => "Time (days)", :log_HBsAg => "log₁₀(HBsAg) IU/mL",
                           group=:id => nonnumeric) *
                    visual(Lines, alpha=0.3, linewidth=0.5)
        draw!(fig[1, 1], plt_hbsag; axis = (title = "HBsAg Dynamics",))

        # Add LOQ line
        ax1 = contents(fig[1, 1])[1]
        hlines!(ax1, [log10(0.05)], color = :red, linestyle = :dash)
    end

    # Viral load panel using AOG
    if :log_V in names(dynamics_df)
        plt_viral = data(sample_df) *
                    mapping(:time => "Time (days)", :log_V => "log₁₀(HBV DNA) copies/mL",
                           group=:id => nonnumeric) *
                    visual(Lines, alpha=0.3, linewidth=0.5)
        draw!(fig[2, 1], plt_viral; axis = (title = "Viral Load Dynamics",))

        # Add LOQ line
        ax2 = contents(fig[2, 1])[1]
        hlines!(ax2, [log10(25)], color = :red, linestyle = :dash)
    end

    Label(fig[0, :], title, fontsize = 16)

    return fig
end

#=============================================================================
# HBV Population Dynamics Plots
=============================================================================#

"""
    plot_hbv_population_dynamics(
        dynamics_df::DataFrame;
        biomarkers::Vector{Symbol} = [:log_HBsAg, :log_V, :log_ALT, :log_E],
        stratify_by::Union{Symbol,Nothing} = :outcome,
        show_loq::Bool = true,
        time_unit::Symbol = :days,
        title::String = "HBV Infection Dynamics"
    ) -> Figure

Create population-level dynamics plot with median and CI bands for HBV biomarkers.

Shows 2x2 panel layout for up to 4 biomarkers, with optional stratification
by infection outcome (acute vs chronic) or treatment arm.

# Arguments
- `dynamics_df`: DataFrame with time series data (from simulate_hbv_dynamics)
- `biomarkers`: Biomarker columns to plot (max 4)
- `stratify_by`: Column to stratify by (e.g., :outcome, :treatment)
- `show_loq`: Whether to show LOQ threshold lines
- `time_unit`: Display time in :days or :weeks
- `title`: Overall figure title

# Returns
Makie Figure with 2x2 panel layout
"""
function plot_hbv_population_dynamics(
    dynamics_df::DataFrame;
    biomarkers::Vector{Symbol} = [:log_HBsAg, :log_V, :log_ALT, :log_E],
    stratify_by::Union{Symbol,Nothing} = :outcome,
    show_loq::Bool = true,
    time_unit::Symbol = :days,
    title::String = "HBV Infection Dynamics"
)
    # Limit to 4 biomarkers
    biomarkers = biomarkers[1:min(4, length(biomarkers))]
    n_biomarkers = length(biomarkers)

    # Determine layout
    nrows = n_biomarkers <= 2 ? 1 : 2
    ncols = n_biomarkers <= 2 ? n_biomarkers : 2

    fig = Figure(size = (400 * ncols, 350 * nrows + 50))

    # Get stratification groups
    if !isnothing(stratify_by) && hasproperty(dynamics_df, stratify_by)
        groups = sort(unique(dynamics_df[!, stratify_by]))
    else
        groups = [nothing]
        stratify_by = nothing
    end

    # Time conversion
    time_divisor = time_unit == :weeks ? 7.0 : 1.0
    time_label = time_unit == :weeks ? "Time (weeks)" : "Time (days)"

    for (i, biomarker) in enumerate(biomarkers)
        row = div(i - 1, 2) + 1
        col = mod(i - 1, 2) + 1

        # Get biomarker label
        ylabel = get(HBV_BIOMARKER_LABELS, biomarker, string(biomarker))

        ax = Axis(fig[row, col],
            xlabel = time_label,
            ylabel = ylabel,
            title = string(biomarker)
        )

        # Plot each group
        for group in groups
            if isnothing(stratify_by)
                group_df = dynamics_df
                group_color = :blue
                group_label = "All"
            else
                group_df = @subset(dynamics_df, cols(stratify_by) .== group)
                group_color = get(HBV_OUTCOME_COLORS, group, colorant"#1f77b4")
                group_label = string(group)
            end

            # Skip if no data for this group
            if nrow(group_df) == 0
                continue
            end

            # Compute summary statistics
            summary = @chain group_df begin
                @groupby(:time)
                @combine(
                    :median = median(cols(biomarker)),
                    :q05 = quantile(cols(biomarker), 0.05),
                    :q25 = quantile(cols(biomarker), 0.25),
                    :q75 = quantile(cols(biomarker), 0.75),
                    :q95 = quantile(cols(biomarker), 0.95)
                )
                @transform(:time_plot = :time ./ time_divisor)
                @orderby(:time)
            end

            # Plot 90% CI band
            band!(ax, summary.time_plot, summary.q05, summary.q95,
                  color = (group_color, 0.2))

            # Plot IQR band
            band!(ax, summary.time_plot, summary.q25, summary.q75,
                  color = (group_color, 0.3))

            # Plot median line
            lines!(ax, summary.time_plot, summary.median,
                   color = group_color, linewidth = 2, label = group_label)
        end

        # Add LOQ threshold lines
        if show_loq && haskey(HBV_LOQ_THRESHOLDS, biomarker)
            hlines!(ax, [HBV_LOQ_THRESHOLDS[biomarker]],
                    color = :red, linestyle = :dash, linewidth = 1)
        end

        # Add legend only to first panel
        if i == 1 && !isnothing(stratify_by)
            axislegend(ax, position = :rt)
        end
    end

    Label(fig[0, :], title, fontsize = 16)

    return fig
end

"""
    plot_hbv_population_dynamics_aog(
        dynamics_df::DataFrame;
        biomarkers::Vector{Symbol} = [:log_HBsAg, :log_V, :log_ALT, :log_E],
        stratify_by::Union{Symbol,Nothing} = :outcome,
        show_loq::Bool = true,
        time_unit::Symbol = :days,
        title::String = "HBV Infection Dynamics"
    ) -> Figure

AlgebraOfGraphics version: Create population-level dynamics plot with median and CI bands.

Uses AOG faceting for multi-panel layout and layers for summary statistics.
Note: AOG doesn't natively support ribbon/band plots, so this uses a hybrid approach
with AOG for line plots and CairoMakie for bands.
"""
function plot_hbv_population_dynamics_aog(
    dynamics_df::DataFrame;
    biomarkers::Vector{Symbol} = [:log_HBsAg, :log_V, :log_ALT, :log_E],
    stratify_by::Union{Symbol,Nothing} = :outcome,
    show_loq::Bool = true,
    time_unit::Symbol = :days,
    title::String = "HBV Infection Dynamics"
)
    # Limit to 4 biomarkers
    biomarkers = biomarkers[1:min(4, length(biomarkers))]

    # Time conversion
    time_divisor = time_unit == :weeks ? 7.0 : 1.0
    time_label = time_unit == :weeks ? "Time (weeks)" : "Time (days)"

    # Stack data into long format for AOG
    long_df = stack(dynamics_df[:, vcat([:id, :time], stratify_by !== nothing ? [stratify_by] : Symbol[], biomarkers)],
                    biomarkers, variable_name=:biomarker, value_name=:value)
    long_df[!, :time_plot] = long_df.time ./ time_divisor

    # Compute summary statistics by time, biomarker, and stratification
    group_cols = stratify_by !== nothing ? [:time_plot, :biomarker, stratify_by] : [:time_plot, :biomarker]

    summary_df = combine(groupby(long_df, group_cols)) do gdf
        (median = median(gdf.value),
         q05 = quantile(gdf.value, 0.05),
         q25 = quantile(gdf.value, 0.25),
         q75 = quantile(gdf.value, 0.75),
         q95 = quantile(gdf.value, 0.95))
    end
    sort!(summary_df, :time_plot)

    # Create the median line plot using AOG
    if stratify_by !== nothing
        plt = data(summary_df) *
              mapping(:time_plot => time_label, :median => "Value",
                     color=stratify_by => string(stratify_by),
                     layout=:biomarker) *
              visual(Lines, linewidth=2)
    else
        plt = data(summary_df) *
              mapping(:time_plot => time_label, :median => "Value",
                     layout=:biomarker) *
              visual(Lines, linewidth=2)
    end

    # Draw using AOG faceting
    fig = draw(plt;
        axis = (ylabel = "Value",),
        figure = (title = title,),
        facet = (linkxaxes = :minimal, linkyaxes = :none)
    )

    # Add bands manually to each axis (AOG doesn't support bands natively)
    for (i, biomarker) in enumerate(biomarkers)
        ax = contents(fig.figure[div(i-1, 2)+1, mod(i-1, 2)+1])[1]

        biomarker_summary = @subset(summary_df, :biomarker .== string(biomarker))

        if stratify_by !== nothing
            groups = unique(biomarker_summary[!, stratify_by])
            for group in groups
                group_summary = @subset(biomarker_summary, cols(stratify_by) .== group)
                group_color = get(HBV_OUTCOME_COLORS, group, colorant"#1f77b4")

                band!(ax, group_summary.time_plot, group_summary.q05, group_summary.q95,
                      color = (group_color, 0.2))
                band!(ax, group_summary.time_plot, group_summary.q25, group_summary.q75,
                      color = (group_color, 0.3))
            end
        else
            band!(ax, biomarker_summary.time_plot, biomarker_summary.q05, biomarker_summary.q95,
                  color = (:blue, 0.2))
            band!(ax, biomarker_summary.time_plot, biomarker_summary.q25, biomarker_summary.q75,
                  color = (:blue, 0.3))
        end

        # Add LOQ lines
        if show_loq && haskey(HBV_LOQ_THRESHOLDS, biomarker)
            hlines!(ax, [HBV_LOQ_THRESHOLDS[biomarker]], color = :red, linestyle = :dash, linewidth = 1)
        end
    end

    return fig
end

"""
    plot_hbv_natural_history(
        dynamics_df::DataFrame;
        biomarkers::Vector{Symbol} = [:log_HBsAg, :log_V, :log_ALT, :log_E],
        title::String = "HBV Natural History: Acute vs Chronic"
    ) -> Figure

Plot natural history (untreated) HBV dynamics showing acute vs chronic outcomes.

Specialized wrapper around plot_hbv_population_dynamics for natural history
visualization with appropriate defaults.
"""
function plot_hbv_natural_history(
    dynamics_df::DataFrame;
    biomarkers::Vector{Symbol} = [:log_HBsAg, :log_V, :log_ALT, :log_E],
    title::String = "HBV Natural History: Acute vs Chronic"
)
    return plot_hbv_population_dynamics(
        dynamics_df;
        biomarkers = biomarkers,
        stratify_by = :outcome,
        show_loq = true,
        time_unit = :days,
        title = title
    )
end

"""
    plot_hbv_treatment_response(
        dynamics_df::DataFrame;
        biomarkers::Vector{Symbol} = [:log_HBsAg, :log_V],
        show_phases::Bool = true,
        phase_times::Union{NamedTuple,Nothing} = nothing,
        title::String = "HBV Treatment Response"
    ) -> Figure

Plot HBV treatment response dynamics with optional phase markers.

# Arguments
- `dynamics_df`: DataFrame with time series data
- `biomarkers`: Biomarker columns to plot
- `show_phases`: Whether to show treatment phase vertical lines
- `phase_times`: NamedTuple with phase timing (from get_phase_times)
- `title`: Plot title
"""
function plot_hbv_treatment_response(
    dynamics_df::DataFrame;
    biomarkers::Vector{Symbol} = [:log_HBsAg, :log_V],
    show_phases::Bool = true,
    phase_times::Union{NamedTuple,Nothing} = nothing,
    title::String = "HBV Treatment Response"
)
    fig = plot_hbv_population_dynamics(
        dynamics_df;
        biomarkers = biomarkers,
        stratify_by = hasproperty(dynamics_df, :treatment) ? :treatment : nothing,
        show_loq = true,
        time_unit = :days,
        title = title
    )

    # Add phase markers if requested and times provided
    if show_phases && !isnothing(phase_times)
        for ax in fig.content
            if ax isa Axis
                add_treatment_phase_markers!(ax, phase_times)
            end
        end
    end

    return fig
end

"""
    add_treatment_phase_markers!(
        ax::Axis,
        phase_times::NamedTuple;
        label_phases::Bool = false
    )

Add vertical lines marking treatment phase transitions to an axis.

# Arguments
- `ax`: Makie Axis to add markers to
- `phase_times`: NamedTuple from get_phase_times() with phase boundaries
- `label_phases`: Whether to add phase labels (experimental)
"""
function add_treatment_phase_markers!(
    ax::Axis,
    phase_times::NamedTuple;
    label_phases::Bool = false
)
    # Phase transition times
    phase_colors = Dict(
        :untreated => :gray,
        :nuc_background => :blue,
        :treatment => :green,
        :off_treatment => :orange
    )

    # Add vertical lines at phase boundaries
    if haskey(phase_times, :untreated)
        vlines!(ax, [phase_times.untreated.stop],
                color = :black, linestyle = :dash, linewidth = 1)
    end

    if haskey(phase_times, :nuc_background) && phase_times.nuc_background.stop > phase_times.nuc_background.start
        vlines!(ax, [phase_times.nuc_background.stop],
                color = :black, linestyle = :dash, linewidth = 1)
    end

    if haskey(phase_times, :treatment)
        vlines!(ax, [phase_times.treatment.stop],
                color = :black, linestyle = :solid, linewidth = 1)
    end

    if haskey(phase_times, :off_treatment)
        vlines!(ax, [phase_times.off_treatment.stop],
                color = :black, linestyle = :solid, linewidth = 1)
    end
end

"""
    plot_hbv_biomarker_panel(
        dynamics_df::DataFrame,
        biomarker::Symbol;
        stratify_by::Union{Symbol,Nothing} = :outcome,
        time_unit::Symbol = :days,
        show_individual::Bool = false,
        n_individual::Int = 20,
        title::String = ""
    ) -> Figure

Create a single-panel plot for one HBV biomarker.

# Arguments
- `dynamics_df`: DataFrame with time series data
- `biomarker`: Single biomarker to plot
- `stratify_by`: Column to stratify by
- `time_unit`: :days or :weeks
- `show_individual`: Whether to show individual patient trajectories
- `n_individual`: Number of individual trajectories to show
- `title`: Plot title
"""
function plot_hbv_biomarker_panel(
    dynamics_df::DataFrame,
    biomarker::Symbol;
    stratify_by::Union{Symbol,Nothing} = :outcome,
    time_unit::Symbol = :days,
    show_individual::Bool = false,
    n_individual::Int = 20,
    title::String = ""
)
    fig = Figure(size = (700, 500))

    time_divisor = time_unit == :weeks ? 7.0 : 1.0
    time_label = time_unit == :weeks ? "Time (weeks)" : "Time (days)"
    ylabel = get(HBV_BIOMARKER_LABELS, biomarker, string(biomarker))

    if isempty(title)
        title = ylabel
    end

    ax = Axis(fig[1, 1],
        xlabel = time_label,
        ylabel = ylabel,
        title = title
    )

    # Get stratification groups
    if !isnothing(stratify_by) && hasproperty(dynamics_df, stratify_by)
        groups = sort(unique(dynamics_df[!, stratify_by]))
    else
        groups = [nothing]
        stratify_by = nothing
    end

    for group in groups
        if isnothing(stratify_by)
            group_df = dynamics_df
            group_color = :blue
            group_label = "All"
        else
            group_df = @subset(dynamics_df, cols(stratify_by) .== group)
            group_color = get(HBV_OUTCOME_COLORS, group, colorant"#1f77b4")
            group_label = string(group)
        end

        if nrow(group_df) == 0
            continue
        end

        # Show individual trajectories if requested
        if show_individual
            unique_ids = unique(group_df.id)
            sample_ids = unique_ids[1:min(n_individual, length(unique_ids))]
            for id in sample_ids
                patient_df = @subset(group_df, :id .== id)
                lines!(ax, patient_df.time ./ time_divisor, patient_df[!, biomarker],
                       color = (group_color, 0.2), linewidth = 0.5)
            end
        end

        # Compute and plot population summary
        summary = @chain group_df begin
            @groupby(:time)
            @combine(
                :median = median(cols(biomarker)),
                :q05 = quantile(cols(biomarker), 0.05),
                :q25 = quantile(cols(biomarker), 0.25),
                :q75 = quantile(cols(biomarker), 0.75),
                :q95 = quantile(cols(biomarker), 0.95)
            )
            @transform(:time_plot = :time ./ time_divisor)
            @orderby(:time)
        end

        band!(ax, summary.time_plot, summary.q05, summary.q95, color = (group_color, 0.2))
        band!(ax, summary.time_plot, summary.q25, summary.q75, color = (group_color, 0.3))
        lines!(ax, summary.time_plot, summary.median,
               color = group_color, linewidth = 2, label = group_label)
    end

    # LOQ threshold
    if haskey(HBV_LOQ_THRESHOLDS, biomarker)
        hlines!(ax, [HBV_LOQ_THRESHOLDS[biomarker]],
                color = :red, linestyle = :dash, linewidth = 1, label = "LOQ")
    end

    if !isnothing(stratify_by)
        axislegend(ax, position = :rt)
    end

    return fig
end

"""
    plot_hbv_biomarker_panel_aog(
        dynamics_df::DataFrame,
        biomarker::Symbol;
        stratify_by::Union{Symbol,Nothing} = :outcome,
        time_unit::Symbol = :days,
        show_individual::Bool = false,
        n_individual::Int = 20,
        title::String = ""
    ) -> Figure

AlgebraOfGraphics version: Create a single-panel plot for one HBV biomarker.

Uses AOG for line plots with group aesthetic, with CairoMakie bands overlaid.
"""
function plot_hbv_biomarker_panel_aog(
    dynamics_df::DataFrame,
    biomarker::Symbol;
    stratify_by::Union{Symbol,Nothing} = :outcome,
    time_unit::Symbol = :days,
    show_individual::Bool = false,
    n_individual::Int = 20,
    title::String = ""
)
    time_divisor = time_unit == :weeks ? 7.0 : 1.0
    time_label = time_unit == :weeks ? "Time (weeks)" : "Time (days)"
    ylabel = get(HBV_BIOMARKER_LABELS, biomarker, string(biomarker))

    if isempty(title)
        title = ylabel
    end

    # Prepare data with time_plot column
    plot_df = copy(dynamics_df)
    plot_df[!, :time_plot] = plot_df.time ./ time_divisor

    fig = Figure(size = (700, 500))

    # Show individual trajectories using AOG if requested
    if show_individual
        # Sample patients
        unique_ids = unique(plot_df.id)
        sample_ids = unique_ids[1:min(n_individual, length(unique_ids))]
        sample_df = @subset(plot_df, :id .∈ Ref(sample_ids))

        if stratify_by !== nothing && hasproperty(plot_df, stratify_by)
            indiv_plt = data(sample_df) *
                        mapping(:time_plot, biomarker,
                               group=:id => nonnumeric, color=stratify_by) *
                        visual(Lines, alpha=0.2, linewidth=0.5)
        else
            indiv_plt = data(sample_df) *
                        mapping(:time_plot, biomarker, group=:id => nonnumeric) *
                        visual(Lines, alpha=0.2, linewidth=0.5)
        end

        draw!(fig[1, 1], indiv_plt;
              axis = (xlabel = time_label, ylabel = ylabel, title = title))
    else
        ax = Axis(fig[1, 1], xlabel = time_label, ylabel = ylabel, title = title)
    end

    ax = contents(fig[1, 1])[1]

    # Compute and plot population summary with bands (hybrid approach)
    if stratify_by !== nothing && hasproperty(plot_df, stratify_by)
        groups = sort(unique(plot_df[!, stratify_by]))
        for group in groups
            group_df = @subset(plot_df, cols(stratify_by) .== group)
            group_color = get(HBV_OUTCOME_COLORS, group, colorant"#1f77b4")

            summary = @chain group_df begin
                @groupby(:time_plot)
                @combine(
                    :median = median(cols(biomarker)),
                    :q05 = quantile(cols(biomarker), 0.05),
                    :q25 = quantile(cols(biomarker), 0.25),
                    :q75 = quantile(cols(biomarker), 0.75),
                    :q95 = quantile(cols(biomarker), 0.95)
                )
                @orderby(:time_plot)
            end

            band!(ax, summary.time_plot, summary.q05, summary.q95, color = (group_color, 0.2))
            band!(ax, summary.time_plot, summary.q25, summary.q75, color = (group_color, 0.3))
            lines!(ax, summary.time_plot, summary.median,
                   color = group_color, linewidth = 2, label = string(group))
        end
        axislegend(ax, position = :rt)
    else
        summary = @chain plot_df begin
            @groupby(:time_plot)
            @combine(
                :median = median(cols(biomarker)),
                :q05 = quantile(cols(biomarker), 0.05),
                :q25 = quantile(cols(biomarker), 0.25),
                :q75 = quantile(cols(biomarker), 0.75),
                :q95 = quantile(cols(biomarker), 0.95)
            )
            @orderby(:time_plot)
        end

        band!(ax, summary.time_plot, summary.q05, summary.q95, color = (:blue, 0.2))
        band!(ax, summary.time_plot, summary.q25, summary.q75, color = (:blue, 0.3))
        lines!(ax, summary.time_plot, summary.median, color = :blue, linewidth = 2)
    end

    # LOQ threshold
    if haskey(HBV_LOQ_THRESHOLDS, biomarker)
        hlines!(ax, [HBV_LOQ_THRESHOLDS[biomarker]],
                color = :red, linestyle = :dash, linewidth = 1)
    end

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
    plot_calibration_result_aog(
        original_values::AbstractVector,
        calibrated_values::AbstractVector,
        target;
        title::String = "MILP Calibration Result"
    ) -> Figure

AlgebraOfGraphics version: Compare original, calibrated, and target distributions.

Uses AOG with faceting for stacked histogram panels.
"""
function plot_calibration_result_aog(
    original_values::AbstractVector,
    calibrated_values::AbstractVector,
    target;
    title::String = "MILP Calibration Result"
)
    # Combine data into long format
    combined_df = DataFrame(
        value = vcat(original_values, calibrated_values),
        source = vcat(
            fill("Original (n=$(length(original_values)))", length(original_values)),
            fill("Calibrated (n=$(length(calibrated_values)))", length(calibrated_values))
        )
    )

    # Create histogram layer
    plt = data(combined_df) *
          mapping(:value => string(target.variable_name)) *
          histogram(bins=length(target.percentages), normalization=:probability) *
          mapping(row=:source => sorter(["Original (n=$(length(original_values)))",
                                         "Calibrated (n=$(length(calibrated_values)))"]))

    fig = draw(plt;
        axis = (ylabel = "Percentage (%)",),
        figure = (title = title,)
    )

    # Add target as separate panel using CairoMakie
    ax_target = Axis(fig.figure[3, 1],
        xlabel = string(target.variable_name),
        ylabel = "Percentage (%)",
        title = "Target Distribution"
    )
    barplot!(ax_target, 1:length(target.percentages), target.percentages,
             color = (:orange, 0.7))

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

"""
    plot_pareto_front_aog(
        pareto_points::Vector;
        optimal_idx::Union{Int, Nothing} = nothing,
        title::String = "Pareto Front: VPs vs Distribution Error"
    ) -> Figure

AlgebraOfGraphics version: Plot Pareto front for MILP calibration.

Uses AOG scatter and lines with point highlighting for optimal.
"""
function plot_pareto_front_aog(
    pareto_points::Vector;
    optimal_idx::Union{Int, Nothing} = nothing,
    title::String = "Pareto Front: VPs vs Distribution Error"
)
    # Build DataFrame from pareto points
    df = DataFrame(
        n_selected = [p.n_selected for p in pareto_points],
        mean_error = [p.mean_error for p in pareto_points],
        point_type = fill("Pareto", length(pareto_points))
    )

    # Mark optimal point if specified
    if !isnothing(optimal_idx) && optimal_idx <= length(pareto_points)
        df[optimal_idx, :point_type] = "Optimal"
    end

    # Create plot with scatter and line
    plt_scatter = data(df) *
                  mapping(:n_selected => "Number of Selected VPs",
                         :mean_error => "Mean Distribution Error (%)",
                         color=:point_type) *
                  visual(Scatter, markersize=10)

    plt_line = data(df) *
               mapping(:n_selected, :mean_error) *
               visual(Lines, color=(:blue, 0.3))

    fig = draw(plt_scatter + plt_line; axis = (title = title,))

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
    plot_gsa_indices_aog(
        gsa_result,
        output::Symbol;
        title::String = "Sensitivity Indices"
    ) -> Figure

AlgebraOfGraphics version: Create bar chart of GSA sensitivity indices.

Uses AOG with dodged bar plot for first and total order indices.
"""
function plot_gsa_indices_aog(
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

    # Build long-format DataFrame
    df = DataFrame(
        parameter = String[],
        index_type = String[],
        index_value = Float64[]
    )

    for p in param_order
        fo_val = filter(r -> r.parameter == p, fo)[1, :index]
        to_val = filter(r -> r.parameter == p, to)[1, :index]
        push!(df, (string(p), "First Order (S₁)", fo_val))
        push!(df, (string(p), "Total Order (Sₜ)", to_val))
    end

    # Create AOG plot with dodged bars
    plt = data(df) *
          mapping(:parameter => sorter(string.(param_order)) => "Parameter",
                 :index_value => "Sensitivity Index",
                 dodge=:index_type, color=:index_type => "Index Type") *
          visual(BarPlot)

    fig = draw(plt;
        axis = (title = "$title: $output", xticklabelrotation = π/6)
    )

    # Add threshold line
    ax = current_axis()
    hlines!(ax, [0.1], color = :red, linestyle = :dash)

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
    plot_gsa_heatmap_aog(
        gsa_result;
        index_type::Symbol = :total_order,
        title::String = "GSA Heatmap"
    ) -> Figure

AlgebraOfGraphics version: Create heatmap of sensitivity indices across outputs.

Uses AOG's heatmap visual with mapping for output and parameter.
"""
function plot_gsa_heatmap_aog(
    gsa_result;
    index_type::Symbol = :total_order,
    title::String = "GSA Heatmap"
)
    indices_df = index_type == :total_order ? gsa_result.total_order : gsa_result.first_order

    # Convert to string columns for AOG
    df = DataFrame(
        output = string.(indices_df.output),
        parameter = string.(indices_df.parameter),
        index = indices_df.index
    )

    # Create heatmap using AOG
    plt = data(df) *
          mapping(:parameter => "Parameter", :output => "Output", :index => "Sensitivity Index") *
          visual(Heatmap, colormap=:viridis)

    fig = draw(plt;
        axis = (title = title, xticklabelrotation = π/4)
    )

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

"""
    plot_gsa_comparison_aog(
        gsa_results::Dict,
        output::Symbol;
        title::String = "GSA Treatment Comparison"
    ) -> Figure

AlgebraOfGraphics version: Compare GSA results across different treatments/conditions.

Uses AOG with dodged bars for treatment comparison.
"""
function plot_gsa_comparison_aog(
    gsa_results::Dict,
    output::Symbol;
    title::String = "GSA Treatment Comparison"
)
    treatments = collect(keys(gsa_results))
    first_result = gsa_results[first(treatments)]
    params = first_result.parameters

    # Build long-format DataFrame
    df = DataFrame(
        parameter = String[],
        treatment = String[],
        index_value = Float64[]
    )

    for treatment in treatments
        to = filter(row -> row.output == output, gsa_results[treatment].total_order)
        for p in params
            idx_val = filter(r -> r.parameter == p, to)[1, :index]
            push!(df, (string(p), string(treatment), idx_val))
        end
    end

    # Create AOG plot with dodged bars
    plt = data(df) *
          mapping(:parameter => "Parameter", :index_value => "Total Order Index",
                 dodge=:treatment, color=:treatment => "Treatment") *
          visual(BarPlot)

    fig = draw(plt;
        axis = (title = title, xticklabelrotation = π/4)
    )

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

"""
    quick_hist_aog(values::AbstractVector; title::String = "Distribution", bins::Int = 30) -> Figure

AlgebraOfGraphics version: Create quick histogram for exploration.

Uses simple AOG histogram pattern.
"""
function quick_hist_aog(values::AbstractVector; title::String = "Distribution", bins::Int = 30)
    df = DataFrame(value = values)
    plt = data(df) * mapping(:value => "Value") * histogram(bins=bins)
    fig = draw(plt; axis = (ylabel = "Count", title = title))
    return fig
end

"""
    quick_scatter_aog(x::AbstractVector, y::AbstractVector;
                      xlabel::String = "x", ylabel::String = "y",
                      title::String = "Scatter Plot") -> Figure

AlgebraOfGraphics version: Create quick scatter plot for exploration.

Uses simple AOG scatter pattern.
"""
function quick_scatter_aog(x::AbstractVector, y::AbstractVector;
                           xlabel::String = "x", ylabel::String = "y",
                           title::String = "Scatter Plot")
    df = DataFrame(x = x, y = y)
    plt = data(df) *
          mapping(:x => xlabel, :y => ylabel) *
          visual(Scatter, markersize=5, alpha=0.5)
    fig = draw(plt; axis = (title = title,))
    return fig
end

#=============================================================================
# Exports
=============================================================================#

export ISCT_THEME, TREATMENT_COLORS, set_isct_theme!

# HBV-specific constants
export HBV_OUTCOME_COLORS, HBV_BIOMARKER_LABELS, HBV_LOQ_THRESHOLDS

# Vpop distribution plots (CairoMakie)
export plot_parameter_distributions, plot_parameter_correlations, plot_vpop_comparison

# Vpop distribution plots (AlgebraOfGraphics)
export plot_parameter_distributions_aog, plot_parameter_correlations_aog, plot_vpop_comparison_aog

# VCT result plots (CairoMakie)
export plot_tumor_dynamics, plot_response_waterfall, plot_treatment_comparison
export plot_hbv_dynamics

# VCT result plots (AlgebraOfGraphics)
export plot_tumor_dynamics_aog, plot_response_waterfall_aog, plot_treatment_comparison_aog
export plot_hbv_dynamics_aog

# HBV population dynamics plots (CairoMakie)
export plot_hbv_population_dynamics, plot_hbv_natural_history, plot_hbv_treatment_response
export plot_hbv_biomarker_panel, add_treatment_phase_markers!

# HBV population dynamics plots (AlgebraOfGraphics)
export plot_hbv_population_dynamics_aog, plot_hbv_biomarker_panel_aog

# Calibration plots (CairoMakie)
export plot_calibration_result, plot_pareto_front

# Calibration plots (AlgebraOfGraphics)
export plot_calibration_result_aog, plot_pareto_front_aog

# GSA plots (CairoMakie)
export plot_gsa_indices, plot_gsa_heatmap, plot_gsa_comparison

# GSA plots (AlgebraOfGraphics)
export plot_gsa_indices_aog, plot_gsa_heatmap_aog, plot_gsa_comparison_aog

# Summary and utility (CairoMakie)
export create_isct_summary_figure, save_figure, quick_hist, quick_scatter

# Utility plots (AlgebraOfGraphics)
export quick_hist_aog, quick_scatter_aog
