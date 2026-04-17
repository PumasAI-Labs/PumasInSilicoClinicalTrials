---
name: isct-step4-vpop
description: Generate a correlated virtual population for a Pumas model using Copulas.jl GaussianCopula + SklarDist for step 4 of the Cortés-Ríos 2025 ISCT workflow — sample random effects that preserve the Spearman rank correlations estimated from real data, then `simobs` the vpop for downstream calibration. USE when the user says "generate a virtual population", "vpop", "virtual patients", "copula sampling", "correlated random effects", "preserve parameter correlations", or is writing `GaussianCopula(Rho)` / `SklarDist(copula, (...))`. USE when `@random` declares a mix of marginal families (Normal + LogNormal + LogitNormal etc.), when independent sampling has destroyed a correlation they need, when they're deciding how many virtual patients to generate (driven by downstream MILP yield, not Sobol-style 2^n), and USE the resample-until-valid `findall(!isvalid, sims)` pattern whenever stiff/QSP ODEs cause integration failures. Do NOT reach for a copula when `@random` is all-Normal with a single `MvNormal(Ω)` — just sample from `MvNormal(zeros(k), cor2cov(Rho, s))` instead.
---

# ISCT Step 4 — Copula-Based Virtual Population (Pumas 2.8 API)

## Purpose

Produce a virtual population whose random effects have the Spearman rank correlations estimated from
real patient data *and* the right per-η marginal families.
NLME fits typically report non-zero Spearman correlations between random effects (e.g.
treatment-sensitive tumors grow more slowly → negative correlation between the η's that map to `f`
and `g` via `@pre`); independent sampling destroys these correlations and yields an unrepresentative
vpop.
The Gaussian copula recovers them exactly.
It earns its weight specifically when `@random` declares a *mix* of marginal families —
e.g. a Normal η for a Normal effect, a LogNormal η for a strictly positive effect,
a LogitNormal η for a bounded effect —
because Sklar's theorem lets you specify each marginal independently of the dependence structure.
If every marginal in `@random` is Normal,
`MvNormal` with the right covariance is equivalent and simpler.

Work at the `@random` layer throughout this step.
The `@pre` transformations are applied by `simobs` internally —
do not pre-apply them during sampling.

## Prerequisites from earlier steps

- From step 1 (`isct-step1-model`):
  a Pumas `@model` whose `@random` block declares the marginal distribution of each random effect.
  The `SklarDist` marginal tuple must match those declarations exactly — same family,
  same parameters.
- Estimated population parameters `param` (or `pop_params`), per-η ω values,
  and a **Spearman correlation matrix** `Rho` —
  usually from an NLME fit summary or the paper/dataset the model came from.
- From step 2 (`isct-step2-gsa`): awareness of which parameters dominate the endpoint.
  A vpop whose marginals are too narrow for a dominant parameter bottlenecks step 5's yield.
- If `Rho` is unknown: **do not proceed**.
  Elicit it or derive from a fit before generating the vpop —
  independent sampling is not a safe default.

## Required inputs (confirm with the user before writing code)

- [ ] `Rho` — Spearman correlation matrix, symmetric, positive semi-definite,
      rows/columns ordered to match `@random`.
- [ ] The marginal distributions declared in `@random` (families + parameters).
      For standard NLME with `η ~ MvNormal(Ω)`,
      that's a tuple of `Normal(0, ω_i)` with `ω_i = sqrt(Ω[i, i])`.
- [ ] `param` / `pop_params` — fixed-effect values to simulate under.
- [ ] `nvps` — target number of virtual patients (2,000 for tutorial-scale TB; typically 2,000–10,
      000, driven by step 5 yield, not Sobol-style power-of-2).
- [ ] The `Subject` template (id, covariates, observation times).

## Core skeleton — mixed-family marginals (the copula earns its keep)

Use this pattern when `@random` declares per-η distributions from different families, e.g.:

```julia
@random begin
    η_f ~ Normal(0, ω_f)        # Normal effect
    η_g ~ LogNormal(0, ω_g)     # strictly positive effect
    η_p ~ LogitNormal(0, ω_p)   # bounded effect
end
```

Match the `SklarDist` marginals to the `@random` declarations exactly:

```julia
using Copulas, Distributions, Pumas

copula = GaussianCopula(Rho)
correlated_randeffs_dist = SklarDist(
    copula,
    (
        Normal(0, ω_f),          # mirrors @random η_f
        LogNormal(0, ω_g),       # mirrors @random η_g
        LogitNormal(0, ω_p),     # mirrors @random η_p
    ),
)

patients = [Subject(; id, covariates = (; treatment = true)) for id in 1:nvps]
correlated_randeffs = rand(correlated_randeffs_dist, nvps)   # (neta × nvps)
vrandeffs = [(; η_f = c[1], η_g = c[2], η_p = c[3]) for c in eachcol(correlated_randeffs)]

sims = simobs(model, patients, param, vrandeffs; simulate_error = false)
```

`@pre` is evaluated inside `simobs` and maps those η's to individual parameters —
don't apply `@pre` manually during sampling.

## Shortcut — all-Normal `@random` with `η ~ MvNormal(Ω)`

This is the tutorials' case.
A Gaussian copula with Normal marginals produces the same samples as `MvNormal` directly,
so skip the copula:

```julia
using Distributions, StatsBase

η_vec = [ω_f, ω_g, ω_k]                                          # standard deviations from Ω
Σ = cor2cov(Rho, η_vec)                                          # correlation + SDs → covariance
η_samples = rand(MvNormal(zeros(length(η_vec)), Σ), nvps)        # η is mean-zero in NLME
vrandeffs = [(; η) for η in eachcol(η_samples)]

sims = simobs(model, patients, param, vrandeffs; simulate_error = false)
```

Both skeletons produce an equivalent vpop when the marginals are all Normal —
the tutorials choose the copula path for pedagogical continuity with the HBV case,
but `MvNormal` is the simpler answer when it applies.

## Resample-until-valid (stiff/QSP models)

For stiff ODEs (HBV pattern), wrap the simulation
so failed integrations are replaced rather than propagated:

```julia
vrandeffs = map(patients) do _
    (; η = rand(correlated_randeffs_dist))   # or draw from MvNormal if applicable
end
sims = simobs(hbv_model, patients, hbv_params, vrandeffs; simulate_error = false, obstimes)

while true
    invalid = findall(!isvalid, sims)
    isempty(invalid) && break
    pop = @view(patients[invalid])
    vrands = @view(vrandeffs[invalid])
    map!(vrands, pop) do _
        (; η = rand(correlated_randeffs_dist))
    end
    sims[invalid] = simobs(hbv_model, pop, hbv_params, vrands; simulate_error = false, obstimes)
end
```

## Validation

Compare observed Spearman correlations to `Rho`, and per-η ω to expected,
using the generated parameters post-`@pre`:

```julia
using StatsBase

vpop = postprocess(sims) do gen, _
    (; f = only(unique(gen.f)), g = only(unique(gen.g)), k = only(unique(gen.k)))
end |> DataFrame

observed_corr = corspearman(Matrix(vpop[:, [:f, :g, :k]]))
# @. abs(observed_corr - Rho) should be small — decreases with larger nvps
```

Comparing Spearman correlations computed on the parameters (`f`, `g`,
`k`) to `Rho` sampled on the η scale is valid **only because** the `@pre` transformations here are
monotonic in each η.
A non-monotonic `@pre` would break this equivalence;
if the user's `@pre` has any non-monotonic mapping,
compare Spearman on the η samples directly instead.

## Decision points

- **Copula or `MvNormal`?**
  If `@random` declares `η ~ MvNormal(Ω)` (every marginal Normal),
  `MvNormal(zeros, cor2cov(Rho, s))` is the simpler path.
  Reach for `GaussianCopula` + `SklarDist` when `@random` mixes marginal families.
- **Correlation matrix source.**
  Prefer Spearman from the NLME fit.
  Spearman is invariant under **monotonic** transforms,
  so the same `Rho` remains valid through `@pre` as long
  as each η enters its parameter monotonically —
  which is true for the standard transforms (`θ * exp(η)`,
  `logistic(logit(θ) + η)`) but not for arbitrary ones.
  Pearson is not invariant even under monotonic transforms — translate before using.
- **Number of VPs.**
  Not a power of 2 — step 5's yield decides the right size.
  Start at 2,000 for TB-like simple models, 500–1,000 for HBV-scale QSP (slower to simulate),
  and scale up by 5× if step 5 is INFEASIBLE or selects too few.
- **Resample-until-valid vs filter-out.**
  For stiff models, resampling preserves `nvps` and keeps the correlation quality near the tails;
  filtering silently shrinks the vpop
  and biases it away from the region the integrator struggles with.
- **`simulate_error = false`** is not optional —
  residual noise contaminates the response classification step 5 keys on.
  Turn it on only in step 6 if the trial readout genuinely should include measurement error.
- **Seeding.**
  Seed `rand` upstream if step 6 needs to replay the same vpop.

## Pitfalls

1. **`SklarDist` marginals that don't match `@random`.**
   The tuple families and parameters must mirror the `@random` block exactly;
   anything else silently gives the wrong marginals.
2. **Transforming samples via `@pre` manually.**
   `@pre` runs inside `simobs` — don't pre-apply it when drawing η's or you'll double-transform.
3. **Using Pearson where Spearman is meant.**
   Pearson on the η-scale distorts under monotonic `@pre` transforms;
   Spearman survives monotonic transforms exactly.
4. **Assuming Spearman survives *any* `@pre`.**
   It only survives monotonic ones.
   If `@pre` has a non-monotonic step (rare in pharmacometrics but possible,
   e.g. a Hill function of η crossing a threshold), validate correlations on the η samples,
   not the parameters.
5. **Mis-sized `Rho` or wrong row order.**
   `Rho` rows/columns must match the `SklarDist` tuple order and count.
6. **Reaching for a copula when every marginal is Normal.**
   Use `MvNormal(zeros, cor2cov(Rho, s))` — mean-zero η, `cor2cov` for the covariance,
   no copula needed.
7. **Hand-building the covariance matrix.**
   Use `cor2cov(Rho, s)` rather than `Diagonal(s) * Rho * Diagonal(s)`;
   the helper is clearer and more discoverable.
8. **`simulate_error = true` during vpop generation.**
   Adds residual noise to observables, inflating the category counts step 5 tries to match.
9. **Ignoring integration failures.**
   Without resample-until-valid,
   stiff models silently lose tail-region patients and the correlation degrades most there.
10. **Generating too few VPs.**
    Sampling variance in the observed Spearman correlations is substantial at n = 100; budget n = 2,
    000+ before judging the copula's fidelity.
11. **Dropping η from the vpop DataFrame.**
    Step 5 selects by row; step 6 re-simulates by η.
    Keep enough of each row to reconstruct the Subject.

## Outputs to surface for sign-off

BEFORE handing off to step 5, show the user:

- The **observed-minus-expected Spearman correlation matrix** (`abs(observed_corr - Rho)`),
  which should be small and shrink with `nvps`.
- The **marginals comparison table** (per-η expected vs observed median and ω).
- A **scatter / corner plot** across η pairs (or the transformed parameters they produce) showing
  the correlation structure — the most convincing proof the sampling worked.
- An **independent-sampling scatter overlay** (a quick independent draw at the same nvps) to show
  the correlation matters — the "striking" figure from tb_04.
- If you used resample-until-valid:
  the **count of initial integration failures** and the **retry count**,
  so the user knows how close the model is to its numerical limits.

Ask: **"Does the observed correlation structure match what you expected from the fit?**
**Is N = <nvps> enough for the calibration yield you need in step 5?"**

Do not silently proceed to step 5.

## When to loop back

- Observed Spearman correlations consistently differ from `Rho` by more than a few percent at large
  N → **stay here**, check the `Rho` definition (symmetric?
  PSD? matched row order to `@random`?) and the marginals' ω values.
- More than ~5% of the initial simulations invalidate → **back to step 1** (`isct-step1-model`),
  the model is too close to a stiffness boundary;
  consider reparameterization or tighter parameter bounds.
- The vpop's empirical outcome distribution can't cover step 5's target (e.g. a target category is
  entirely absent) → **stay here**, widen the marginals (if physiologically defensible) or increase
  `nvps`.
- Step 5 returns INFEASIBLE even after relaxing ε → **stay here**, enlarge the vpop (5×);
  if still infeasible, **back to step 1** — the target may be unreachable under this model.
- Step 6 CIs are too wide → either **stay here** to enlarge the vpop or go to step 5 to loosen ε;
  the diagnosis is "selected vpop too small", not a step 6 problem.

## What this step feeds into

- Step 5 (`isct-step5-calibration`) consumes the **`vpop` DataFrame** with simulated outcomes
  classified into the target's categories (attach `vpop.response` via `cut(...)` before calling step
  5).
- Step 6 (`isct-step6-vct`) consumes the **selected vpop's η** to re-simulate each patient under
  every treatment arm, reusing the same η across arms for paired comparisons.

## Tutorial reference

- `tutorials/tb_04_copula_vpop_tutorial.qmd` — TB (3 parameters, single `f-g` correlation):
  full six-step copula workflow (lines 325–437),
  independent-vs-copula visual comparison (lines 511–561),
  Spearman vs Pearson rationale (lines 563–577).
  Uses all-Normal `@random`, so an `MvNormal`-with-`cor2cov` shortcut would produce equivalent
  samples.
- `tutorials/hbv_04_copula_vpop_tutorial.qmd` — HBV (9 η's, 8 non-zero correlations):
  9×9 Spearman matrix (lines 398–411), `SklarDist` with all-Normal marginals (lines 436–447),
  `conv_E` as an independent row/column (lines 414–421).
  Same caveat — all-Normal `@random`.
- `tutorials/hbv_05_milp_calibration_tutorial.qmd` lines 422–451 —
  the canonical resample-until-valid loop for stiff QSP models.
