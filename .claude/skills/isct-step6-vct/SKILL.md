---
name: isct-step6-vct
description: Run the virtual clinical trial for step 6 of the Cortés-Ríos 2025 ISCT workflow — use the step-5 calibrated vpop to `simobs` across treatment arms, summarize continuous endpoints with AlgebraOfGraphics visuals (time-course bands, spaghetti plots, waterfalls), and compute response-rate confidence intervals with `HypothesisTests.BinomialTest`. USE when the user says "run the virtual trial", "simulate the arms", "VCT", "virtual clinical trial", "response rate confidence interval", "waterfall plot", "spaghetti plot of the vpop", "Clopper-Pearson CI for my vpop", or is designing `obstimes` / treatment-vs-control covariates / endpoint classifications post-calibration. USE when they need to decide obstimes density, arm covariate plumbing, or how to report trial outcomes in a format that matches a clinical readout. Do NOT use for the calibration step that precedes it (that's `isct-step5-calibration`).
---

# ISCT Step 6 — Virtual Clinical Trial Simulation & Analysis (Pumas 2.8 API)

## Purpose

Take the step-5 calibrated virtual population
and run it through each treatment arm of the trial being emulated,
producing endpoint readouts comparable to a real clinical trial —
time-course plots with median and inter-quartile bands, individual spaghetti trajectories,
response-rate tables with confidence intervals,
and waterfall plots of per-patient change from baseline.
The same η vector is reused across arms so comparisons are paired.
This step's outputs are the regulatory-adjacent deliverable:
what a user shows a stakeholder to argue the model reproduces a trial signal.

## Prerequisites from earlier steps

- From step 5 (`isct-step5-calibration`):
  the **calibrated vpop DataFrame** (`ilp_vpop` / `milp_vpop` / the bi-objective selection),
  with a column carrying each subject's η vector so `simobs` can re-simulate the exact patients.
- From step 1 (`isct-step1-model`):
  the `@model` with covariate plumbing that encodes each trial arm.
  For a two-arm trial, a binary `treatment` covariate; for multi-arm HBV-style trials,
  time-varying combinations of `NA` and `IFN`.
- If the calibrated vpop is missing η per subject: **do not proceed**.
  Hand back to step 5 to ensure η is retained in the selected rows.

## Required inputs (confirm with the user before writing code)

- [ ] `calibrated_vpop::DataFrame` with per-row η
      and any identifiers needed to reconstruct `Subject`s.
- [ ] `obstimes` — observation schedule matching the clinical readout (weekly
      for the TB 18-week trial; monthly/weekly phase-specific for HBV).
- [ ] **Arm definitions** — for each arm, the covariate values (`treatment = true/false`;
      `NA = [...]`, `IFN = [...]` as time-varying vectors for HBV) and any time-varying schedule.
- [ ] **Endpoint definition** — classification thresholds
      (`cut(..., [0.14, 0.7, 1.0]; labels = ["CR", "PR", "SD", "PD"])`) that must match whatever
      step 5 calibrated against.
- [ ] **CI method and level** — `BinomialTest` (Clopper-Pearson exact)
      for categorical response rates;
      `OneSampleTTest` / `OneSampleHotellingT2Test` etc. from `HypothesisTests`
      for continuous summaries.
- [ ] Which plots to produce (time-course band, spaghetti, waterfall, bar chart with CI).

## Core skeleton — two-arm simulation with paired η

```julia
using Pumas, DataFramesMeta, HypothesisTests, AlgebraOfGraphics, CairoMakie

obstimes = 7 * (0:18)                               # weekly, 18 weeks

# Paired: same η across arms — subject ids must be aligned
treatment_pop = [
    Subject(; id, covariates = (; treatment = true))
        for id in 1:nrow(calibrated_vpop)
]
control_pop = [
    Subject(; id, covariates = (; treatment = false))
        for id in 1:nrow(calibrated_vpop)
]
vrandeffs = [(; row.η) for row in eachrow(calibrated_vpop)]

treatment_results = simobs(
    model, treatment_pop, param, vrandeffs;
    obstimes, simulate_error = false
) |> DataFrame
control_results = simobs(
    model, control_pop, param, vrandeffs;
    obstimes, simulate_error = false
) |> DataFrame

all_results = vcat(
    treatment_results, control_results;
    source = "arm" => ["Treatment", "Control"]
)
```

Multi-arm HBV pattern using time-varying covariates:

```julia
treatment_arms = DataFrame(
    arm = ["Control", "NA", "IFN", "NA + IFN"],
    NA = [false, true, false, true],
    IFN = [false, false, true, true],
)

vct_sims = mapreduce(vcat, eachrow(treatment_arms)) do arm
    pop = map(selected_vps) do id
        Subject(;
            id,
            covariates = (;
                NA = [false, true, arm.NA, false],
                IFN = [false, false, arm.IFN, false],
            ),
            covariates_time = vct_covariates_time
        )
    end
    sims = simobs(
        hbv_model, pop, hbv_params, calibrated_vrandeffs;
        obstimes, simulate_error = false
    )
    filter!(isvalid, sims)                           # drop failed integrations
    DataFrame(sims) |> df -> (df.arm .= arm.arm; df)
end
```

## Endpoint analysis — response rates with confidence intervals

```julia
# Classify the readout at trial end (thresholds must match step 5)
final = @rsubset(all_results, :time == maximum(obstimes))
final_classify = @transform final :response = cut(
    :tumor_size, [0.14, 0.7, 1.0];
    labels = ["CR", "PR", "SD", "PD"], extend = true,
)

# Response rates + Clopper-Pearson 95% CIs
response_ci = @chain final_classify begin
    @groupby :arm
    @transform :npatients = length(:arm)
    @by [:arm, :response] @astable begin
        n = only(unique(:npatients))
        k = length(:response)
        :rate = 100 * k / n
        :ci = round.(100 .* confint(BinomialTest(k, n); level = 0.95); sigdigits = 3)
    end
end
```

## Visualization — four load-bearing plots

Time-course with IQR band and reference lines:

```julia
time_summary = @by all_results [:time, :arm] @astable begin
    q05, q25, q50, q75, q95 = quantile(:tumor_size, (0.05, 0.25, 0.5, 0.75, 0.95))
    :median = q50; :q25 = q25; :q75 = q75; :q05 = q05; :q95 = q95
end

specs = data(time_summary) * mapping(
    :time => (t -> t / 7) => "Time [week]",
    color = :arm => sorter("Treatment", "Control") => "Arm",
) * (
    mapping(:q25, :q75) * visual(Band; alpha = 0.3) +
        mapping(:median) * visual(Lines)
)
refs = mapping(
    [1.0, 0.14],
    linestyle = ["Baseline", "Complete response"] => "Reference"
) *
    visual(HLines)
draw(specs + refs; axis = (; ylabel = "Normalized tumor size"))
```

Spaghetti plot (sampled subset), waterfall plot (per-patient % change sorted),
and bar chart with CI error bars follow the same AoG idiom —
see tb_06 lines 559–697 for the canonical forms.

## Decision points

- **`obstimes` density.**
  Match the clinical readout you're emulating.
  Weekly for TB; phase-specific (monthly baseline / weekly treatment / weekly follow-up) for HBV.
  Over-dense `obstimes` inflates simulation time without adding information;
  under-dense obscures transient dynamics.
- **Arm covariate design.**
  For time-varying regimens (HBV multi-phase),
  build `covariates_time` and a vector-valued covariate per subject; for simple on/off (TB),
  a scalar binary covariate on `Subject(; covariates = (; treatment = Bool))` is enough.
- **Endpoint thresholds.**
  Must match step 5's calibration thresholds exactly — use the same `cut` break points and labels,
  or the calibrated match is silently inconsistent with what you report.
- **`simulate_error` for the VCT.**
  Turn on *only* if the trial readout genuinely includes measurement noise the user wants reflected
  in the CIs.
  Off is the default (clean mechanistic trajectories).
- **CI method.**
  `BinomialTest` for categorical response rates (exact Clopper-Pearson).
  For continuous endpoints (mean ALT change, median viral load log drop),
  use `OneSampleTTest` or bootstrap, depending on distributional assumptions.
- **How many plots.**
  Four is the canonical set: time-course band, spaghetti, waterfall, response-rate bar with CIs.
  Each answers a different stakeholder question —
  budget the space for all four if the audience is regulatory.

## Pitfalls

1. **Different η per arm.**
   If you resample η for each arm, comparisons stop being paired.
   Always build `vrandeffs` once from the calibrated vpop and pass it to every arm's `simobs` call.
2. **`obstimes` that don't include the step-5 readout time.**
   Your per-arm response rates will drift from the calibration match
   because `cut` runs at the wrong timepoint.
3. **Endpoint thresholds that drift between step 5 and step 6.**
   Hardcode them in one place or reference the same symbol.
4. **Forgetting `filter!(isvalid, sims)` for stiff models.**
   HBV patients can still invalidate here even if step 4 resampled successfully,
   because the arm covariate mix changes the integration regime.
   Filter after `simobs` in every arm.
5. **Reusing the uncalibrated vpop by mistake.**
   Double-check you're handing `simobs` the calibrated subset's η's, not the full step-4 vpop.
6. **Claiming a significant treatment effect from point estimates alone.**
   Always report CIs; the bar chart with error bars from `BinomialTest.confint` is the minimum.
7. **Log-scale y-axes without zero-handling.**
   Spaghetti plots with `yscale = log10` will error on zeros — clip or offset before plotting.

## Outputs to surface for sign-off

BEFORE declaring the VCT complete, show the user:

- The **response-rate table with 95% CIs** (`response_ci`), one row per arm × response category.
- The **time-course band plot** with median + IQR per arm
  and reference lines at baseline / response threshold.
- A **spaghetti plot** on a sampled subset (~30 patients), color-coded by response classification.
- A **waterfall plot** of per-patient percent change from baseline, sorted and arm-faceted.
- A **calibrated-vs-uncalibrated time-course comparison** —
  simulate the uncalibrated step-4 vpop through the same arms and show both panels side by side.
  This is the single best evidence that calibration mattered.
- A short **summary sentence per endpoint**: e.g. "Treatment arm complete-response rate:
  30% (95% CI 22%–39%); control arm 8% (95% CI 4%–14%)."

Ask: **"Do these rates and trajectories match the trial you're emulating?**
**Anything surprising that would send us back to the calibration or vpop step?"**

Do not silently declare the workflow done.

## When to loop back

- Response-rate CIs are wider than the user needs (e.g. spanning 0) → the
  **calibrated vpop is too small**.
  Go **back to step 5** to loosen ε,
  or **back to step 4** to enlarge the vpop before re-calibrating.
- The trial fails to reproduce a known effect
  that individual patients in the calibrated vpop *should* show → calibration may have pruned
  responders.
  **Back to step 5**; inspect the calibrated subset's parameter distribution against the full vpop.
- Individual patient trajectories diverge from what a single-patient step-1 simulation produces
  under the same covariates → **covariate plumbing bug in the model**.
  **Back to step 1**; check `@covariates`, `@pre`, and time-varying covariate construction.
- Many integrations fail (`isvalid == false`) in a specific arm →
  that arm's covariate combination hits a stiffness boundary.
  **Back to step 1** to reparameterize, or **stay here** if the failure rate is small and filtered.
- Endpoint rates don't match step 5's calibration target → **threshold drift**.
  Check that `cut(...)` thresholds in this step match the ones step 5 used to classify responses.
- The bar chart with CIs tells a story,
  but the time-course shows something inconsistent (e.g. high CR rate
  but low median drop) → **mixture in the vpop**: responders and non-responders are both present.
  Use the spaghetti plot to confirm and reconsider
  whether the single-summary endpoint is the right story.

## What this step feeds into

- The user's decision or deliverable — regulatory filing, protocol design, stakeholder brief.
  There is no step 7; loop-back targets are the earlier ISCT steps.
- If the user is iterating on a protocol (different dosing, longer follow-up,
  different patient population),
  the outputs feed back to **step 6 with new arm covariates**
  or **step 4 with a different vpop spec**.

## Tutorial reference

- `tutorials/tb_06_vct_simulation_tutorial.qmd` — canonical two-arm VCT:
  paired simulation (lines 361–395),
  time-course summary + band plot with references (lines 397–465),
  calibrated-vs-uncalibrated comparison (lines 467–557),
  spaghetti plot with response coloring (lines 559–598),
  response classification + `BinomialTest` CIs (lines 600–665), waterfall (lines 667–697).
- `tutorials/hbv_06_vct_simulation_tutorial.qmd` — HBV four-arm VCT:
  time-varying covariates across untreated/NA/treatment/follow-up phases (lines 569–587),
  functional cure endpoint (HBsAg + HBV DNA composite, lines 603–680),
  `filter!(isvalid, sims)` usage for stiff simulations.
