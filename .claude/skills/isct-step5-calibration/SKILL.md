---
name: isct-step5-calibration
description: Calibrate a virtual population to a target outcome distribution via JuMP + HiGHS — per-category ILP with a uniform ε bound, MILP with a mean-total-error (MTE) bound, or a bi-objective Pareto front using MultiObjectiveAlgorithms.jl — for step 5 of the Cortés-Ríos 2025 ISCT workflow. USE when the user says "calibrate my vpop", "match response rates", "MILP", "ILP", "JuMP calibration", "Pareto front of vpop size vs. error", or is writing `@variable(model, x[1:N], Bin)` / `@objective(model, Max, sum(x))` / `MOA.EpsilonConstraint()`. CRITICAL: USE when calibration is infeasible ("no solution", `assert_is_solved_and_feasible` throws, HiGHS reports INFEASIBLE) — this skill encodes the diagnosis ladder (relax ε → enlarge vpop in step 4 → revisit target / model in step 1). USE when choosing between per-category ILP, MTE-MILP, and bi-objective. Do NOT use for Pumas model fitting (`fit(...)`).
---

# ISCT Step 5 — MILP Calibration (Pumas 2.8 API)

## Purpose

Select a subset of virtual patients from the step-4 vpop whose simulated outcome distribution
matches a target distribution from a real (or target-design) trial, so that downstream virtual
clinical trial simulations (step 6) reflect the population the trial actually enrolled.
The core decision is how strictly to enforce the match — per-category (tight,
every category within ±ε), by mean total error (looser, average across categories within ε),
or traced across a Pareto front.
This step is the usual site of ISCT surprises:
infeasibility here is almost always a signal that the step-4 vpop is too narrow,
that the target is unreachable under the step-1 model, or that ε is unrealistically tight.

## Prerequisites from earlier steps

- From step 4 (`isct-step4-vpop`):
  a `vpop::DataFrame` of virtual patients with per-patient simulated outcomes attached (e.g.
  `vpop.response` for categorical readouts, or the continuous variable that gets binned with
  `cut(...)`).
- A **target distribution** DataFrame with a category column (e.g. `response`)
  and a percentage column (e.g. `pct`) that sums to 100.
  Source: the trial you're trying to match, or a study design specification.
- If either is missing: **do not proceed**.
  Hand back to step 4 (for the vpop) or clarify the target with the user.

## Required inputs (confirm with the user before writing code)

- [ ] `vpop::DataFrame` with simulated outcomes already classified into the target's categories.
- [ ] `target_distribution::DataFrame` with matching category labels and percentages.
- [ ] `epsilon` — tolerance, same ε symbol across formulations
      but *different meaning* (see skeletons below).
- [ ] Formulation choice: ILP / MTE-MILP / bi-objective (default:
      explore bi-objective first if unsure; pick ε from the Pareto front).
- [ ] Solver time limit (`set_time_limit_sec(model, 60.0)` is the tutorial default;
      raise for large vpops).
- [ ] Reproducibility: set a random seed upstream if later steps should be replayable.

## Core skeletons (three variants)

### Variant A — per-category ILP (`|q_c − p_c| ≤ ε` for every category)

```julia
using JuMP, HiGHS

epsilon = 0.05
model = Model(HiGHS.Optimizer)
set_silent(model); set_attribute(model, "presolve", "on")
set_time_limit_sec(model, 60.0)

@variable(model, x[1:nrow(vpop)], Bin)
N_total = sum(x)
@objective(model, Max, N_total)      # as many VPs as possible, subject to match

for row in eachrow(target_distribution)
    p_c = row.pct / 100
    N_c = sum(x[vpop.response .== row.response])
    @constraint(model, N_c <= (p_c + epsilon) * N_total)
    @constraint(model, N_c >= (p_c - epsilon) * N_total)
end

optimize!(model)
assert_is_solved_and_feasible(model)  # throws on INFEASIBLE — see loop-back section
ilp_selected = findall(xi -> value(xi) == 1, x)
ilp_vpop = vpop[ilp_selected, :]
```

### Variant B — MTE-MILP (mean total error bound, allows inter-category trade-offs)

```julia
model_milp = Model(HiGHS.Optimizer)
set_silent(model_milp); set_attribute(model_milp, "presolve", "on")
set_time_limit_sec(model_milp, 60.0)

@variable(model_milp, x_milp[1:nrow(vpop)], Bin)
@variable(model_milp, δ[1:nrow(target_distribution)] >= 0)   # per-category count slack
N_total = sum(x_milp)
@objective(model_milp, Max, N_total)

for (δ_c, row) in zip(δ, eachrow(target_distribution))
    p_c = row.pct / 100
    N_c = sum(x_milp[vpop.response .== row.response])
    @constraint(model_milp, δ_c >= N_c - p_c * N_total)
    @constraint(model_milp, δ_c >= p_c * N_total - N_c)
end
# sum(δ) = C · MTE · N_total, so MTE ≤ ε ⇔ sum(δ) ≤ C·ε·N_total
@constraint(model_milp, sum(δ) <= nrow(target_distribution) * epsilon * N_total)

optimize!(model_milp)
assert_is_solved_and_feasible(model_milp)
```

MTE-MILP's feasible set strictly contains the per-category ILP's at the same ε,
so it always selects at least as many VPs —
at the cost of allowing an individual category to exceed ε in absolute error.

### Variant C — bi-objective Pareto front (explore before committing to ε)

```julia
using MultiObjectiveAlgorithms as MOA

moa_model = Model(() -> MOA.Optimizer(HiGHS.Optimizer))
set_attribute(moa_model, MOA.Algorithm(), MOA.EpsilonConstraint())
set_attribute(moa_model, MOA.SolutionLimit(), 50)
set_silent(moa_model)

@variable(moa_model, y[1:nrow(vpop)], Bin)
@variable(moa_model, Δ >= 0)                       # max absolute count displacement
N_total = sum(y)
@objective(moa_model, Min, [-N_total, Δ])          # bi-objective: maximize N, minimize Δ

for row in eachrow(target_distribution)
    p_c = row.pct / 100
    N_c = sum(y[vpop.response .== row.response])
    @constraint(moa_model, N_c - p_c * N_total <= Δ)
    @constraint(moa_model, p_c * N_total - N_c <= Δ)
end

optimize!(moa_model)
assert_is_solved_and_feasible(moa_model)
```

**Recovering the ε_max-Pareto front.**
MOA returns a Δ-Pareto front; ε_max = Δ / N_total is what you actually care about.
Every ε_max-Pareto-optimal solution is also Δ-Pareto-optimal,
so you can post-filter by sweeping the Δ-front in descending N_total
and keeping solutions whose ε_max strictly improves on the running minimum:

```julia
moa_df = map(1:result_count(moa_model)) do i
    obj = objective_value(moa_model; result = i)
    nselected = round(Int, -obj[1]); delta = obj[2]
    (; nselected, max_error = nselected > 0 ? delta / nselected : NaN)
end |> DataFrame
sort!(moa_df, [order(:nselected; rev = true), :max_error])
moa_df.pareto_optimal .= false
moa_df.pareto_optimal[begin] = true
best = moa_df.max_error[begin]
for row in eachrow(moa_df)
    if row.max_error < best
        row.pareto_optimal = true
        best = row.max_error
    end
end
```

## Decision points

- **Formulation.**
  Per-category ILP is appropriate when a regulator
  or clinical question demands every category be tight.
  MTE-MILP is the right default for most ISCT workflows (flexibility, same or more VPs).
  Bi-objective when you don't yet know what ε is affordable — let the Pareto front tell you.
- **ε choice.**
  If you committed up-front, 0.05 is the tutorial default (5% per-category or 5% MTE).
  If you ran the bi-objective first, pick the ε at the knee of the Pareto curve.
- **Category definition.**
  The vpop must be pre-classified into the target's categories.
  For continuous outcomes, use `cut(...)` from CategoricalArrays with the same thresholds step 6
  will use to report results.
  Misaligned thresholds silently produce inconsistent response rates later.
- **Time limit.**
  60 s for tutorial-scale (~2k VPs).
  HiGHS scales well; raise to a few minutes for ~10k VPs.
  A `TIME_LIMIT` exit is **not** infeasibility — you just ran out of time.
  Raise it or accept the incumbent.
- **Seed / reproducibility.**
  The binary decisions are deterministic given the solver state,
  but the upstream vpop sampling is not.
  Seed step 4 if you need step 6 to replay.

## Pitfalls

1. **`assert_is_solved_and_feasible` throwing — don't conflate INFEASIBLE with TIME_LIMIT.**
   HiGHS distinguishes; check `termination_status(model)` before reaching for the diagnosis ladder.
2. **Forgetting that MTE-MILP's ε is *not* a per-category guarantee.**
   Individual categories can exceed ε in absolute error;
   if any regulator downstream cares about per-category error,
   compute and report `maximum(abs, comparison_df.pct_diff)` alongside MTE.
3. **Reporting Δ as if it were the metric of interest.** It isn't — ε_max is.
   `max_error = Δ / N_total`.
   This is the single most common bi-objective mistake.
4. **Using the Δ-Pareto front unfiltered.**
   Δ-dominance implies ε_max-dominance but *not* the other way;
   there are Δ-Pareto points that are ε_max-dominated.
   Always post-filter per the snippet above.
5. **Category labels that don't match between `vpop.response` and `target_distribution.response`.**
   `leftjoin` with `@coalesce(:pct_vpop, 0.0)` will silently treat missing categories
   as 0% in the vpop and let the solver find a "solution" that's actually ignoring the missing
   category.
6. **Reusing the uncalibrated vpop in step 6.**
   Hand step 6 the filtered `ilp_vpop` / `milp_vpop` / selected bi-objective subset,
   never the full step-4 vpop.

## Outputs to surface for sign-off

BEFORE handing off to step 6, show the user:

- A **before/after comparison table**: target percentage, uncalibrated vpop percentage,
  calibrated vpop percentage, absolute difference — per category.
- A **before/after comparison chart**:
  grouped bars with `row = :source => ["Original", "Calibrated"]` from the tutorial pattern.
- **Summary metrics**: selected vpop size N, maximum absolute per-category error `ε_max`,
  mean total error MTE.
- If you ran bi-objective: the **Pareto front scatter** (ε_max vs N_total) with the selected
  solution highlighted.
- The **identifier list** of selected VPs (either indices or the calibrated DataFrame row IDs)
  so step 6 can re-simulate the right subjects.

Ask: **"Does this calibrated distribution match the clinical trial you're emulating?**
**Are you happy with N = <size> selected patients at ε_max = <value>, or do you want to trade off
one for the other?"**

Do not silently proceed to step 6.

## When to loop back

Use this ladder when calibration is infeasible or disappointing:

- **INFEASIBLE at reasonable ε** → first **stay here**, relax ε (double it,
  or move to MTE-MILP from per-category ILP); if still infeasible at ε ≳ 0.15,
  the per-category targets are too sharp for the vpop as generated.
- **Still INFEASIBLE after relaxing** → **back to step 4** (`isct-step4-vpop`),
  enlarge the vpop (5× is a reasonable first bump).
  If a target category is entirely absent from the vpop —
  check the per-category counts before blaming the solver.
- **INFEASIBLE even with a 10× larger vpop** → **back to step 1** (`isct-step1-model`),
  the target distribution is unreachable under the current model/assumption set.
  Negotiate the target with the user or reconsider a structural assumption.
- **Selected vpop too small for step 6 statistical power** (e.g. a category has only 2 patients) →
  **stay here**, loosen ε; or **back to step 4** to enlarge.
- **Calibrated parameter distribution looks unrealistic** (e.g. the ILP picked only the extreme
  tails of the vpop) → **back to step 4** — the unconstrained vpop was too narrow to give the solver
  a realistic interior region to work with.
- **Per-category error acceptable in MTE-MILP but one specific category is far off** →
  **stay here**, switch to per-category ILP for that category only (hybrid is fine: MTE constraint
  plus a single per-category bound).
- **Solver times out (`TIME_LIMIT`, not `INFEASIBLE`)** → raise `set_time_limit_sec`,
  or reduce vpop size first.

## What this step feeds into

- Step 6 (`isct-step6-vct`) consumes the **calibrated vpop** (`ilp_vpop` / `milp_vpop` / the
  bi-objective selection), specifically its per-patient random effects, to re-simulate under each
  treatment arm.
- If the bi-objective Pareto front was traced,
  also pass the **chosen (N, ε_max)** pair so step 6's power calculations know the context.

## Tutorial reference

- `tutorials/tb_05_milp_calibration_tutorial.qmd` — canonical three-variant reference:
  ILP (lines 440–608), bi-objective + ε_max Pareto recovery (lines 610–799),
  MTE-MILP (lines 800–977).
- `tutorials/hbv_05_milp_calibration_tutorial.qmd` —
  HBV calibration on baseline HBsAg bins from the Everest trial (lines 490–588);
  also the canonical spot for the step-4 resample-until-valid pattern
  that guarantees the vpop is clean before calibration.
