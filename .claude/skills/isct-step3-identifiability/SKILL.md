---
name: isct-step3-identifiability
description: Check structural identifiability of a Pumas model's parameters with StructuralIdentifiability.jl (local and global) for step 3 of the Cortés-Ríos 2025 ISCT workflow — translate the Pumas ODE system via `model.sys`, specify measurement scenarios with `@variables x(t)` and `~` measurement equations, and classify each parameter as globally / locally / non-identifiable under each observation setup. USE when the user says "identifiability", "identifiable parameters", "can I estimate these", "StructuralIdentifiability", "global vs local identifiability", or when GSA has narrowed the parameter set and estimability needs confirming before moving to calibration. CRITICAL: USE whenever the model's ODE RHS contains non-rational terms (Hill exponents that aren't integers, fractional powers, conditionals, threshold switches) — this skill encodes the auxiliary-state rational reformulation trick (the HBV `H` state for `(V+S)^0.486`); see `references/non-rational-reformulation.md` for the full derivation. Do NOT use for practical / numerical identifiability from real data — that's profile likelihoods, not this skill.
---

# ISCT Step 3 — Structural Identifiability (Pumas 2.8 API)

## Purpose

Determine, purely from the model's equations and the planned measurement setup,
which parameters can in principle be uniquely estimated.
Structural identifiability is a *necessary* condition for meaningful estimation:
if a parameter is non-identifiable under your measurement plan,
no amount of data will ever pin it down,
and trying to estimate it will produce plausible-looking fits with arbitrary parameter values.
The three verdicts are **globally identifiable** (unique value),
**locally identifiable** (finitely many values — still ambiguous),
and **non-identifiable** (infinitely many — no amount of data helps).
The output reshapes step 1 (reparameterize or change the measurement plan)
and scopes step 5 (don't calibrate against targets
that depend on non-identifiable degrees of freedom).

## Prerequisites from earlier steps

- From step 1 (`isct-step1-model`):
  a Pumas `@model` whose `@dynamics` is expressible as a rational ODE system —
  polynomial RHS or quotient of polynomials in the state variables and parameters.
  If it isn't, see the sidecar below and reformulate before coming here.
- From step 2 (`isct-step2-gsa`): the parameter ranking.
  Don't waste SI time on parameters GSA already shows have no effect at the endpoint —
  focus on the ones that actually matter.
- If `@dynamics` contains non-integer exponents, `ifelse` / threshold switches,
  or any non-rational term: **do not proceed**.
  Read `references/non-rational-reformulation.md`
  and return to step 1 to introduce the auxiliary state.

## Required inputs (confirm with the user before writing code)

- [ ] The Pumas `@model` and its extracted ODE system (`ode_system = model.sys`).
- [ ] The **measurement scenarios** —
      one or more sets of observation equations
      (`@variables y(t); measured = [y ~ ode_system.state]`).
- [ ] Which parameters are **fixed** (known from literature) vs **estimated** —
      fixed ones should be declared measured so the algorithm treats them as known.
- [ ] Whether to run `assess_identifiability` (global — comprehensive,
      slow) or `assess_local_identifiability` (local — rank-based, fast screening).
      For models with ≳ 5 parameters, screen local first;
      run global only on the parameters that pass.
- [ ] Whether to follow up with `find_identifiable_functions` to discover identifiable parameter
      combinations when individual parameters are non-identifiable.
- [ ] Whether inputs (doses) affect identifiability — if so,
      they must be defined as inputs in a dedicated ModelingToolkit ODE system (see tutorial callout
      at tb_03 lines 302–313).

## Core skeleton — single measurement scenario

```julia
using Pumas, StructuralIdentifiability, ModelingToolkit
import Logging

ode_system = tumor_burden_model.sys                           # extract MTK system

t = ModelingToolkit.independent_variable(ode_system)
@variables Nt(t)                                              # time-dependent measurement

measured_total_only = [Nt ~ ode_system.N_sens + ode_system.N_res]   # `~` = measurement equation

result_total = assess_identifiability(
    ode_system;
    measured_quantities = measured_total_only,
    loglevel = Logging.Warn,                                  # suppress verbose logs
)
# result_total :: Dict{Num, Symbol} with values :globally / :locally / :nonidentifiable
```

## Core skeleton — multiple scenarios, compared side-by-side

```julia
@variables Nt(t) N_sens_obs(t) N_res_obs(t)

scenarios = [
    "Total tumor only" => [Nt ~ ode_system.N_sens + ode_system.N_res],
    "Both populations" => [
        N_sens_obs ~ ode_system.N_sens,
        N_res_obs ~ ode_system.N_res,
    ],
]

results = Dict(
    name => assess_identifiability(
            ode_system;
            measured_quantities = eqs,
            loglevel = Logging.Warn
        )
        for (name, eqs) in scenarios
)
```

## Fixed parameters declared as measured (HBV pattern)

When the model has structural parameters fixed from literature (HBV has ~22),
the algorithm must know they are *known*, not unknown-to-be-estimated.
Declare a symbolic output per fixed parameter and add to `measured_quantities`:

```julia
@variables y_p_V y_d_V y_r_X      # one per fixed parameter

measured_fixed = [
    y_p_V ~ ode_system.p_V,
    y_d_V ~ ode_system.d_V,
    y_r_X ~ ode_system.r_X,
    # … one entry per fixed parameter
]

result = assess_identifiability(
    ode_system;
    measured_quantities = vcat(scenario_A_measurements, measured_fixed),
    loglevel = Logging.Warn,
)
```

## Local screening before global (for large systems)

```julia
local_result = assess_local_identifiability(
    ode_system;
    measured_quantities = measured,
    loglevel = Logging.Warn,
)
# Run assess_identifiability only on the subset of parameters that passed locally —
# globally is a strictly stronger property, so locally-false → globally-false already.
```

## Finding identifiable combinations when individuals aren't

```julia
find_identifiable_functions(
    ode_system;
    measured_quantities = measured_total_only,
    loglevel = Logging.Warn,
)
# Returns symbolic expressions — e.g. {g - k, g * k}. If individual g and k are
# non-identifiable but their sum/product are, reparameterize step 1 around the
# combinations.
```

## Decision points

- **Measurement scenario(s).**
  Which biomarkers are clinically feasible and affordable?
  Run the analysis for each candidate plan;
  choose the minimal plan that makes the parameters you care about globally identifiable.
  For ISCT, the scenarios are usually: what a routine clinic measures, what the trial adds,
  what a lab-only biomarker would add.
- **Local vs global.**
  Local is cheap (rank of a Jacobian) and good for screening;
  global is expensive (Gröbner-basis-like machinery) but definitive.
  For > 5 parameters: local first, global on the survivors.
- **Initial conditions as parameters.**
  Parameters that live *only* in initial conditions are treated specially by the algorithm —
  often returned as non-identifiable even when they're trivially observable.
  Reparameterize to lift them into `@param` with a measurable projection,
  or treat them as known from a first measurement.
- **Fixed parameters handling.**
  Declare every fixed parameter as measured.
  Missing declarations make them look unknown and poison the analysis.
- **When to reparameterize vs re-measure.**
  Non-identifiable + cheap biomarker → add measurement.
  Non-identifiable + expensive biomarker → reparameterize around identifiable combinations (`g - k`,
  `g * k`) and live with the reduced-dimensional estimand.

## Pitfalls

1. **Calling `assess_identifiability(model)` directly on the Pumas model.**
   It expects an MTK `ODESystem` — always pass `model.sys`.
2. **Using `==` instead of `~` in measurement equations.**
   The `~` operator (from ModelingToolkit / Symbolics) is what the algorithm parses
   as a measurement; `==` is Boolean equality and will silently confuse things.
3. **Defining `@variables Nt` without `(t)`.**
   Measurements must be time-dependent; omitting the `(t)` gives you a parameter symbol,
   not a trajectory symbol.
4. **Non-rational RHS terms.**
   Non-integer Hill exponents (`x^0.486`), fractional powers, `ifelse` / threshold switches —
   all unsupported.
   Reformulate via the auxiliary-state trick before retrying;
   see `references/non-rational-reformulation.md`.
5. **Forgetting to add fixed parameters to `measured_quantities`.**
   The algorithm treats them as unknowns by default,
   and the analysis comes back implausibly pessimistic.
6. **Interpreting `:locally` as safe to estimate.**
   It means there are finitely many parameter values consistent with the data.
   Estimation can still converge to the wrong one —
   always seed the optimizer with a good initial guess, and validate against known biology.
7. **Running identifiability on parameters GSA says are irrelevant.**
   Wastes time; also, non-identifiable-and-irrelevant is the easy case (just fix them)
   so analysing them doesn't add information.
8. **Skipping identifiability entirely and going straight to calibration.**
   Bad fits on non-identifiable parameters look like reasonable fits on any given dataset —
   you only discover the problem when step 6 gives nonsense,
   or when someone independently replicates with different initial guesses.

## Outputs to surface for sign-off

BEFORE handing off to step 4, show the user:

- The **identifiability table** — rows = parameters (and selected state trajectories),
  columns = measurement scenarios, cells = `:globally` / `:locally` / `:nonidentifiable`.
- A **plain-language verdict per parameter**: "globally identifiable under measurement plan B —
  safe to estimate", "locally identifiable — seed estimation with literature value",
  "non-identifiable — fix to X or reparameterize".
- If any key parameter is non-identifiable and you ran `find_identifiable_functions`:
  the **list of identifiable combinations** and a note on
  which reparameterization would remove the degeneracy.
- The **measurement-plan recommendation** —
  which scenario the user should adopt based on cost/feasibility, and which parameters that unlocks.

Ask: **"Does the measurement plan you can actually run match what this analysis says is needed?**
**If not, which parameters are you willing to fix, and which combinations do you want to
reparameterize around?"**

Do not silently proceed to step 4 with non-identifiable parameters still in the estimand.

## When to loop back

- Key parameter is non-identifiable under every realistic measurement scenario → **back to step 1**
  (`isct-step1-model`) to reparameterize (combine parameters, lift initial conditions into `@param`,
  drop a redundant degree of freedom).
- `assess_identifiability` errors out with a complaint about rational RHS → **back to step 1** to
  apply the auxiliary-state reformulation in `references/non-rational-reformulation.md`.
- Locally identifiable but not globally → **stay here**;
  accept the finite ambiguity but flag it to step 4
  so estimation runs start from a good initial guess.
  For ISCT, this is usually fine because simulation doesn't need point estimation.
- GSA flagged a parameter as influential and SI says non-identifiable → **back to step 1**.
  You cannot estimate what GSA says matters under the current plan —
  either add a biomarker or reparameterize.
- A parameter that only appears in initial conditions comes back non-identifiable → **stay here**,
  check if it's observable at t=0 under the measurement plan.
  If yes, note that the SI algorithm's blindspot doesn't reflect practical estimability; if no,
  lift it into `@param` with an explicit observable.
- Algorithm never returns (Gröbner basis not converging) → **stay here**,
  fall back to `assess_local_identifiability` for screening;
  run global on a restricted parameter subset.

## What this step feeds into

- Step 4 (`isct-step4-vpop`) consumes the **identifiable parameter set** —
  only those become variable η's;
  non-identifiable parameters are fixed to literature values or to reparameterized combinations.
- Step 5 (`isct-step5-calibration`) consumes the **measurement plan** —
  calibration targets should only reference outputs derived from identifiable parameters;
  otherwise calibration is fitting noise in an unidentifiable direction.

## Tutorial reference

- `tutorials/tb_03_structural_identifiability_tutorial.qmd` — TB (3 parameters):
  ODE extraction (line 283), `@variables` + `~` measurement equations (lines 318–355),
  global analysis (lines 363–367), interpreting globally/locally/non (lines 376–427),
  initial-condition caveat for `f` (lines 429–447), `find_identifiable_functions` (lines 449–494).
- `tutorials/hbv_03_structural_identifiability_tutorial.qmd` —
  HBV (9 estimated + 22 fixed parameters): auxiliary-state reformulation (lines 68–117,
  fully derived in the sidecar), `@model` with the added `H` state (lines 120–286),
  measurement scenarios (lines 313–339), declaring fixed parameters as measured (lines 358–410),
  local-first screening (lines 418–443),
  visualization of identifiability across scenarios (lines 478–507).
- `references/non-rational-reformulation.md` (this skill's sidecar) —
  full auxiliary-state derivation for Hill functions with non-integer exponents.
