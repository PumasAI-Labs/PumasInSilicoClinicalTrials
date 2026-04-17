---
name: isct-step1-model
description: Structure a Pumas 2.8 NLME `@model` for step 1 of the Cortés-Ríos 2025 ISCT workflow — pick `@param` domains, `@random` marginals, η-to-parameter transforms in `@pre`, `@covariates` for trial arms, `@dynamics` that downstream steps can consume, and `@observed` / `@derived` endpoints for GSA and fitting. USE when the user is building or refactoring a Pumas model that will be consumed by a later ISCT step (you'll see `@model`, `@param`, `@pre`, `@dynamics`, `@derived`), when an earlier ISCT step (GSA ranking, identifiability failure, MILP infeasibility, VCT covariate bug) is forcing reparameterization, or when they ask "how should I set up my model for ISCT / for a virtual trial / for sensitivity analysis in Pumas". USE especially when their `@dynamics` has a non-rational term (fractional Hill exponent, threshold switch) — flag it here so step 3's identifiability analysis doesn't fail. Do NOT use for generic NLME fitting workflow questions unrelated to ISCT.
---

# ISCT Step 1 — Model Structure for ISCT (Pumas 2.8 API)

## Purpose

The Pumas `@model` is the root of the ISCT workflow:
every later step reads from it (GSA scores its `@observed` / `@derived` outputs,
structural identifiability analyzes `model.sys`, the vpop samples its `@random`,
calibration classifies its simulated outcomes, VCT re-simulates across arms via its `@covariates`).
This step is about *shaping* the model so the downstream steps work —
not about model discovery or fitting.
The choices here that matter most are: domain constraints on `@param`,
marginal families in `@random`, η→parameter maps in `@pre`, covariate plumbing for trial arms,
and whether `@dynamics` is rational (required by step 3).

## Prerequisites from earlier steps

- None — step 1 is the root of the forward path.
  Back-edges from other steps target this skill specifically when reparameterization
  or structural change is needed.

## Required inputs (confirm with the user before writing code)

- [ ] The biological/clinical system being modeled — what states, what observables, what treatments.
- [ ] Parameter values from the literature or a prior fit (typical values, ω's, residual error).
- [ ] The clinical covariates the trial will vary — single binary (`treatment`),
      time-varying combinations (`NA`, `IFN` across phases), dose regimens.
- [ ] The intended endpoints — scalar summaries (`last(tumor_size)`), trajectories (`log10_HBsAg`),
      continuous vs categorical readouts.
- [ ] Any non-rational RHS terms (fractional Hill exponents, threshold switches, `ifelse`) —
      these require reformulation before step 3 can run.

## Core skeleton — the canonical ISCT-ready `@model`

```julia
using Pumas

isct_model = @model begin
    @metadata begin
        desc = "<human-readable description>"
        timeu = u"d"                                  # or u"h", u"wk" — be explicit
    end

    @param begin
        # Typical values — constrain domains realistically
        tvf ∈ RealDomain(lower = 0.0, upper = 1.0)    # bounded fraction
        tvg ∈ RealDomain(lower = 0.0)                 # strictly positive rate
        tvk ∈ RealDomain(lower = 0.0)

        # IIV on the η-scale — PDiagDomain for diagonal, PSDDomain for full covariance
        Ω ∈ PDiagDomain(3)

        # Residual error
        σ ∈ RealDomain(lower = 0.0)
    end

    @random begin
        η ~ MvNormal(Ω)                               # all-Normal — step 4 can use MvNormal directly
        # Alternative: declare per-η families explicitly when mixing LogNormal/LogitNormal etc.
    end

    @covariates treatment                             # arm indicator consumed by @pre / @dynamics

    @pre begin
        # Monotonic η → parameter maps preserve Spearman correlations for step 4
        f = logistic(logit(tvf) + η[1])               # logit-normal: bounded (0,1)
        g = tvg * exp(η[2])                           # log-normal: positive
        k = treatment * tvk * exp(η[3])               # treatment modulates efficacy
    end

    @init begin
        N_sens = f
        N_res = 1 - f
    end

    @dynamics begin
        # Keep the RHS RATIONAL — no fractional exponents, no ifelse, no threshold switches.
        # Non-rational terms break step 3; if unavoidable, see
        # isct-step3-identifiability/references/non-rational-reformulation.md
        N_sens' = -k * N_sens
        N_res' = g * N_res
    end

    @derived begin
        Nt := @. N_sens + N_res                       # `:=` = INTERNAL (not visible to GSA, not observed)
        tumor_size ~ @. Normal(Nt, σ)                 # stochastic observable — fitted; also visible to GSA in 2.8.1+
    end

    @observed begin
        final_tumor = last(tumor_size)                # scalar endpoint — idiomatic for GSA
    end
end
```

## Covariate plumbing for multi-phase trials (HBV pattern)

When arms differ in time-varying covariate schedules (untreated → NA background → NA+IFN →
follow-up), use vector-valued covariates with matching `covariates_time`:

```julia
Subject(;
    id,
    covariates = (;
        NA = [false, true, true, false],
        IFN = [false, false, true, false],
    ),
    covariates_time = [0.0, t_chronic, t_na_suppression_end, t_eot],
)
```

The `@pre` block reads the current covariate value; `@dynamics` consumes whatever `@pre` produces.
Get the time vector right or the phase transitions happen at the wrong instant.

## Validating the model — simulate a typical patient

Use `center_randeffs` to simulate the *center* of each η's declared distribution.
This coincides with `zero_randeffs` only when every `@random` marginal is Normal with mean zero;
for LogNormal / LogitNormal / non-zero-mean Normal marginals,
`zero_randeffs` is not at the center of the distribution.
Default to `center_randeffs`:

```julia
subject = Subject(id = 1, covariates = (; treatment = true), time = [0, 18 * 7])

typical_η = center_randeffs(isct_model, subject, pop_params)
typical_sim = simobs(
    isct_model, subject, pop_params, typical_η;
    obstimes = 0:(7 * 18), simulate_error = false
)
```

For all-Normal `@random` with mean-zero η,
`zero_randeffs` gives the same result and is fine in that narrow case;
`center_randeffs` is the right general-purpose default.

## Decision points

- **Domain constraints on `@param`.**
  Every population parameter gets a physiologically meaningful domain —
  `RealDomain(lower, upper)` for bounded, `RealDomain(lower = 0.0)` for positive.
  `PDiagDomain(n)` for a diagonal Ω;
  `PSDDomain(n)` when the fit estimates off-diagonal correlations.
  Tight domains prevent downstream integrator blow-ups.
- **`@random` marginals.**
  Default is `η ~ MvNormal(Ω)` — all Normal, step 4 can use `MvNormal` directly.
  Declare per-η families (`η_f ~ Normal`, `η_g ~ LogNormal`,
  `η_p ~ LogitNormal`) only if mixed families genuinely match the data;
  this is the regime where step 4's copula earns its weight.
- **η → parameter transforms in `@pre`.**
  Use **monotonic** maps: `logistic(logit(θ) + η)` for bounded parameters in (0,1),
  `θ * exp(η)` for positive ones, `θ + η` for signed.
  Monotonicity means Spearman correlations survive the transform exactly in step 4,
  and GSA ranges can be picked on the η-scale with physical meaning preserved.
- **`@covariates` design.**
  Binary `treatment` for simple two-arm trials; vector-valued time-varying covariates (`NA`,
  `IFN`) plus `covariates_time` for multi-phase regimens.
  Every arm covariate must be plumbed into `@pre` or `@dynamics` —
  declaring it without using it is a silent no-op.
- **`@observed` vs `@derived`.**
  In Pumas 2.8.1+ both are visible to GSA.
  Use `@observed` for scalar summaries (`last(…)`, `maximum(…)`, functional-cure composites);
  use `@derived` for the trajectories that need residual-error distributions for fitting.
  Variables marked internal with `:=` in `@derived` are *not* visible to GSA or fitting.
- **Rational `@dynamics`.**
  Step 3's `StructuralIdentifiability.jl` requires polynomial RHS or polynomial quotients.
  Non-integer Hill exponents, fractional powers, and `ifelse` / threshold switches break it.
  Reformulate via the auxiliary-state trick (see sidecar) *at step 1*, not later.
- **`@init` and initial conditions.**
  Parameters that live only in `@init` are treated differently by SI
  and often come back non-identifiable even when trivially observable.
  If the user plans to estimate an initial-condition parameter,
  lift it to `@param` and observe a projection.

## Pitfalls

1. **Non-rational terms in `@dynamics`.** Breaks step 3.
   Reformulate during step 1 — see
   `../isct-step3-identifiability/references/non-rational-reformulation.md`.
2. **Declaring a covariate in `@covariates` but never consuming it in `@pre` / `@dynamics`.**
   Silent no-op — the arm has no effect, and step 6's arm comparison will look identical to control.
3. **Using `:=` for a quantity you want GSA (step 2) to score.**
   `:=` marks it internal; lift to `@observed` or use `=` in `@derived`.
4. **Non-monotonic `@pre` maps.**
   Breaks Spearman preservation in step 4.
   Stick to `logistic(logit(θ) + η)`, `θ * exp(η)`, `θ + η`, or compositions of monotonic functions.
   `θ + η^2` would *not* be monotonic in η.
5. **Using `zero_randeffs` for a typical-patient simulation when `@random` is non-Normal.**
   Only matches the distribution center for mean-zero Normal marginals;
   use `center_randeffs` by default.
6. **Forgetting `@init` values that satisfy the model's own steady-state** (HBV pattern).
   If the model assumes chronic infection before treatment,
   the initial condition must satisfy the pre-treatment steady state —
   otherwise early-time behaviour is nonsense.
7. **Missing unit declaration (`timeu`).**
   Makes `obstimes` and covariate schedules error-prone across steps.
8. **`Ω` with the wrong domain** (e.g. `RealDomain` instead of `PDiagDomain`/`PSDDomain`).
   Estimation crashes; simulations get negative variance.
9. **Skipping residual-error specification.**
   Step 2 wants residual-error SDs zeroed in `base_params`; step 4 wants `simulate_error = false`;
   step 6 has a choice.
   You need `σ` declared in `@param` to control it explicitly per step.

## Outputs to surface for sign-off

BEFORE handing off to step 2, show the user:

- A **typical-patient simulation** via `center_randeffs` across the planned `obstimes`
  and each arm — confirms the arms are plumbed and the dynamics are qualitatively right.
- A **population simulation** with modest n (~100) under default Ω —
  confirms variability looks sensible and no integration failures at typical η.
- A **clinical-validation check**: nadir time, peak/trough, response rate at readout,
  or whatever endpoint the user will calibrate against in step 5.
  Plot against the expected clinical benchmark.
- A **covariate-effect visualization**: control vs each treatment arm overlaid,
  so the user can confirm the arm covariates modulate the right parameters.
- A **flag for non-rational RHS terms** if any are present,
  with a pointer to the reformulation sidecar.

Ask: **"Does the typical-patient simulation match the clinical behavior you expect?**
**Are the arm differences pointing in the right direction?**
**Any structural change you want to make before we run GSA and identifiability?"**

Do not silently proceed to step 2 on an unvalidated model.

## When to loop back

Step 1 is the root, so the usual direction is forward —
but it receives many back-edges from other steps:

- Step 2 says a parameter explains essentially all variance → simplify the model here (fix
  or drop the dominated-by parameter).
- Step 3 says a key parameter is non-identifiable → reparameterize (combine parameters,
  lift an initial condition, change the measurement plan) here.
- Step 3 errors on non-rational RHS → apply the auxiliary-state reformulation here
  (`references/non-rational-reformulation.md` in the step-3 skill).
- Step 4 sees >5% integration failures → the model is too close to a stiffness boundary;
  reparameterize or tighten domains here.
- Step 5 INFEASIBLE at every ε and vpop size → the target distribution is unreachable under this
  model; revisit a structural assumption here.
- Step 6 arm comparison looks identical to single-patient step-1 behaviour
  or diverges implausibly → covariate plumbing is wrong; fix the `@pre` / `@dynamics` coupling here.

## What this step feeds into

- Step 2 (`isct-step2-gsa`) consumes the **model with `@observed` / `@derived` endpoints**
  and a realistic `base_params` with variances zeroed.
- Step 3 (`isct-step3-identifiability`) consumes **`model.sys`** — requires rational `@dynamics`.
- Step 4 (`isct-step4-vpop`) consumes the **`@random` declaration** —
  its marginals must match the `SklarDist` tuple.
- Step 5 (`isct-step5-calibration`) consumes the **classification thresholds** applied to the
  model's outcomes; they must match step 6's.
- Step 6 (`isct-step6-vct`) consumes the **`@covariates` plumbing** for each arm;
  any covariate declared here must actually modulate dynamics.

## Tutorial reference

- `tutorials/tb_01_model_introduction_tutorial.qmd` — TB canonical form:
  literature-based parameter values (lines 172–189),
  complete `@model` with all blocks (lines 192–260), block-by-block commentary (lines 263+),
  typical-patient vs population simulation (lines 388–469),
  clinical validation of nadir / response (lines 504–576).
- `tutorials/hbv_01_model_introduction_tutorial.qmd` — HBV canonical form:
  complex @model with 12 states (lines 232–421), fixed + estimated parameter split (lines 430–489),
  multi-phase protocol setup (lines 590–615), treatment-mechanism modeling (lines 492–523),
  functional-cure endpoint (lines 772–831).
