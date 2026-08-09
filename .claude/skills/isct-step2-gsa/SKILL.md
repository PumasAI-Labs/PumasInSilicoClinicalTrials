---
name: isct-step2-gsa
description: Run Pumas + GlobalSensitivity.jl global sensitivity analysis (Sobol or eFAST) on an NLME model for step 2 of the Cortés-Ríos 2025 ISCT workflow — rank parameters by first-order and total-order indices, surface interactions, and decide which parameters to carry into structural identifiability and calibration. USE when the user says "sensitivity analysis", "GSA", "Sobol indices", "eFAST", "which parameters matter", "rank my parameters", or is writing `gsa(model, subject, params, …)`. USE when they're choosing sample sizes (mention 2^n), picking GSA endpoints, picking parameter ranges, handling the `constantcoef = (:Ω, :σ)` tuple-of-symbols gotcha, or interpreting total-order minus first-order as interaction evidence. USE when a later ISCT step needs a ranking to prune the parameter set. Do NOT use for local/gradient sensitivity or for NONMEM-style covariate screening.
---

# ISCT Step 2 — Global Sensitivity Analysis (Pumas 2.8 API)

## Purpose

Rank the NLME model's estimated parameters by how much they drive an ISCT-relevant endpoint (e.g.
tumor size at end of treatment, log₁₀ HBsAg at week 48).
The outputs are first-order (Sᵢ) and total-order (Sₜᵢ) Sobol-style indices,
plus an interaction proxy Sₜᵢ − Sᵢ.
Interpret them relatively: near-zero total-order means the parameter does essentially nothing at
this endpoint and is a candidate to fix; near-one first-order means the parameter dominates on its
own; near-one total-order with small first-order means the parameter acts through interactions.
The ranking informs which parameters are worth checking for identifiability (step 3) and
which deserve care in calibration (step 5).

## Prerequisites from earlier steps

- From step 1 (`isct-step1-model`): a Pumas `@model` that exposes GSA endpoints.
  Starting from Pumas 2.8.1, any variable in `@derived` that is part of the simulation output —
  i.e. anything not marked internal with `:=` — is visible to `gsa`.
  `@observed` is still idiomatic for compact scalar summaries (`final_tumor = last(tumor_size)`),
  but it is not required.
- If the model has *only* `:=` internals and no exposed outputs: **do not proceed**.
  Hand back to `isct-step1-model` to add a named endpoint.

## Required inputs (confirm with the user before writing code)

- [ ] The model (e.g. `tumor_burden_model_gsa`, `hbv_model`) with at least one exposed output.
- [ ] Which endpoint(s) to analyze — a scalar (`final_tumor`) or a vector over time (`log10_HBsAg`).
- [ ] `base_params` — fixed-effect values to simulate around,
      with `Ω = Diagonal(zeros(...))` and residual-error SDs set to zero.
- [ ] `p_range_low` and `p_range_high` NamedTuples —
      physiologically plausible bounds for every parameter to vary.
- [ ] Which parameters to hold constant (`constantcoef`) —
      always includes `:Ω` and every residual-error SD.
- [ ] Sample budget — eFAST ≥ 75 samples per factor for exploration,
      Sobol a power of two (`2^10 = 1024` is a reasonable floor).
- [ ] Whether to parallelize (`batch = true`, `ensemblealg = EnsembleThreads()`) —
      worth it for stiff/QSP models.

## Core skeleton

```julia
using GlobalSensitivity, Pumas

subject = Subject(id = 1, covariates = (; treatment = true), time = [0, 18 * 7])

p_range_low = (tvf = 0.05, tvg = 0.0005, tvk = 0.002)
p_range_high = (tvf = 0.95, tvg = 0.005, tvk = 0.05)

base_params = (
    tvf = 0.27, tvg = 0.0013, tvk = 0.0091,
    Ω = Diagonal(zeros(3)),   # deterministic simulations
    σ = 0.0,
)

# eFAST first — cheap exploration
gsa_efast = gsa(
    model, subject, base_params, eFAST(),
    [:final_tumor], p_range_low, p_range_high;
    constantcoef = (:Ω, :σ),              # tuple of SYMBOLS, not a NamedTuple
    samples = 1_000,
)

# Sobol second — publication-grade; samples MUST be a power of two
gsa_sobol = gsa(
    model, subject, base_params, Sobol(),
    [:final_tumor], p_range_low, p_range_high;
    constantcoef = (:Ω, :σ),
    samples = 2^10,
)
```

For stiff/QSP models (HBV pattern) parallelize the ensemble:

```julia
gsa_efast = gsa(
    hbv_model, subject, base_params, eFAST(),
    [:log10_HBsAg], params_lower, params_upper;
    constantcoef = gsa_constantcoef,       # fixed params + all σ terms + :Ω
    samples = 75,
    ensemblealg = EnsembleThreads(),
    batch = true,
)
```

Assemble the ranked table:

```julia
using DataFramesMeta
gsa_first = stack(
    gsa_sobol.first_order, Not(:dv_name);
    variable_name = :parameter, value_name = :first_order
)
gsa_total = stack(
    gsa_sobol.total_order, Not(:dv_name);
    variable_name = :parameter, value_name = :total_order
)
gsa_df = @chain outerjoin(gsa_first, gsa_total; on = [:dv_name, :parameter]) begin
    @rtransform :interaction = :total_order - :first_order
    sort!([:dv_name, order(:total_order; rev = true)])
end
```

## Interpreting the indices (no fixed thresholds)

Read relatively, not against a cutoff:

- **Sₜᵢ ≈ 0** — parameter has essentially no effect on this endpoint.
  Candidate to fix before step 3/5.
- **Sᵢ ≈ 1** — parameter's effect is nearly deterministic on its own;
  estimation should be straightforward.
- **Sₜᵢ ≈ 1 with Sᵢ ≪ Sₜᵢ** — parameter dominates through interactions with others;
  expect identifiability and calibration to be harder.
- **Rank, don't threshold.**
  The interesting question is which parameters rank high at *this* endpoint,
  not whether they cross an arbitrary number.
  The user sets the cutoff in context.
- **Interaction column Sₜᵢ − Sᵢ** is a structural property of the model at that endpoint —
  large values mean a nonlinear response surface, not that the parameter should be dropped.

## Decision points

- **Method.**
  eFAST for cheap exploration and for models with many parameters; Sobol for the final ranking.
  Run both when in doubt — agreement across methods is its own validation.
- **Sample size.**
  eFAST can be as low as 75 samples *per factor* for screening;
  Sobol needs `samples = 2^n` for the quasi-Monte Carlo stratification to work (1024 is a reasonable
  floor, 4096+ for low-signal problems).
- **Parameter ranges.**
  Physiological, not arbitrary.
  For stiff models, derive from the fitted Ω (e.g. `±sqrt.(diag(Ω)) / 3` on the η-scale,
  as hbv_02 does) so the bounds stay inside the simulation-stable region.
- **`constantcoef` content.**
  Always `:Ω` and every residual-error parameter.
  For models with fixed structural parameters (HBV has ~22),
  also include every fixed-parameter name so `gsa` varies only what you want varied.
- **Endpoint choice.**
  A scalar-over-time endpoint (`[:log10_HBsAg]` indexed by time) tells you *when* each parameter
  matters; a single scalar (`[:final_tumor]`) tells you *whether* it matters at the readout time.
  Prefer the former when you can afford it.
- **`@observed` vs `@derived`.** Both work in Pumas 2.8.1+.
  `@observed` reads well for scalar summaries;
  `@derived` outputs (non-`:=`) are fine for trajectories you already defined for the fit.
  Pick whichever keeps the model readable.
- **Parallelization.**
  `ensemblealg = EnsembleThreads(), batch = true` is nearly free for HBV-class models — use it.

## Pitfalls

1. **`constantcoef` must be a tuple of symbols**, not a NamedTuple.
   Correct: `(:Ω, :σ)`.
   Incorrect: `(Ω = Diagonal(zeros(3)), σ = 0.0)`.
   Actual values come from `base_params`.
2. **Sobol with non-power-of-two samples** silently loses stratification
   and inflates index variance.
   Always `samples = 2^n`.
3. **Parameter ranges that cause integration failures** produce `NaN` indices.
   Validate bounds first with a `simobs` at the low and high η tails;
   tighten ranges (or derive from Ω as in hbv_02) before running GSA.
4. **Forgetting to zero `Ω` and residual-error SDs in `base_params`** mixes random-effect noise into
   sensitivity, contaminating Sᵢ and Sₜᵢ.
   Deterministic simulations are the point.
5. **Using `:=` for a variable you actually want GSA to score.**
   `:=` marks it internal; `gsa` can't see it.
   Either drop the `:=` or lift the quantity into `@observed`.
6. **Quoting a hard cutoff for "influential"**.
   There isn't one.
   Rank relatively and let the user decide where to draw the line for this endpoint.
7. **Stopping at one treatment arm.**
   Run GSA per-arm if the covariate structure means control
   and treatment have different sensitivity profiles.

## Outputs to surface for sign-off

BEFORE handing off to step 3 or step 4, show the user:

- The **ranked `gsa_df` DataFrame** (columns `dv_name`, `parameter`, `first_order`, `total_order`,
  `interaction`), sorted by total-order descending.
- A **bar chart** of first-order and interaction (stacked) per parameter,
  faceted by endpoint if you ran multiple.
  No arbitrary threshold line.
- A short **plain-language ranking summary**: parameters near zero,
  parameters that dominate on their own, parameters that matter through interactions.
- For multi-timepoint endpoints:
  a **heatmap or small-multiples chart** of Sₜᵢ across time to show how importance shifts over the
  readout window.

Ask: **"Does this ranking match your mechanistic intuition?**
**Which parameters do you want to carry into the identifiability and calibration steps, and which
are you comfortable fixing?"**

Do not silently proceed to the next step.

## When to loop back

- A single parameter's Sᵢ is ≈ 1 and all others ≈ 0 → **back to step 1** (`isct-step1-model`),
  because the model likely has a redundant parameter you can fix
  or a structural simplification worth making.
- All parameters look similarly ranked with Sₜᵢ far from 0 or 1 → **stay here**,
  the ranges are too narrow to expose the nonlinearity; widen `p_range_*` and rerun.
- Indices contain `NaN` or are unstable across runs → **stay here**,
  either increase samples (especially for Sobol;
  try `2^12`) or tighten ranges to avoid the regions that blow up the integrator.
- A top-ranked parameter turns out non-identifiable in step 3 → **back to step 1** to reparameterize
  (combine parameters, change the measurement plan) — not back to step 2.
- Sensitivity ranking changes qualitatively between control and treatment arms → **stay here**,
  run the analysis per-arm and report both rankings into step 3.

## What this step feeds into

- Step 3 (`isct-step3-identifiability`) consumes the **parameter ranking** to scope the
  identifiability check — don't spend SI time on parameters GSA says do nothing.
- Step 4 (`isct-step4-vpop`) consumes the **Spearman correlation structure** you'll elicit during
  model fitting; the GSA ranking informs which parameters' marginals deserve careful range-setting.

## Tutorial reference

- `tutorials/tb_02_gsa_tutorial.qmd` — TB (toy, 3 parameters):
  canonical eFAST → Sobol workflow (lines 220–470),
  `constantcoef` and 2^n rationale (lines 377–503), multi-timepoint analysis (lines 786–962).
- `tutorials/hbv_02_gsa_tutorial.qmd` — HBV (QSP, 9 parameters):
  deriving bounds from fitted Ω (lines 385–411),
  `batch = true` + `EnsembleThreads()` for stiff systems (lines 430–444),
  tidying time-indexed outputs (lines 454–483).
