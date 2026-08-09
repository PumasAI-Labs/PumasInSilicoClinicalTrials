---
name: isct-workflow
description: Route an in-silico clinical trial (ISCT) workflow for a Pumas 2.8 NLME model end-to-end — the six-step Cortés-Ríos 2025 CPT:PSP loop (model → GSA → structural identifiability → copula vpop → MILP calibration → virtual clinical trial). USE when the user says "apply ISCT", "run a virtual trial", "end-to-end virtual trial", "build a virtual population and calibrate", "implement the Cortés-Ríos workflow", "go through the full ISCT pipeline", or any request that spans more than one of {GSA, identifiability, vpop, calibration, VCT simulation} on a Pumas model. Also USE as the first touch when it's unclear which step the user is on, or when an earlier step's output invalidates a later step and you need to decide where to re-enter the workflow. This skill hands off to the six step-specific `isct-step*` skills; invoke it before picking a specific step unless the user explicitly named the step.
---

# ISCT Workflow Router (Pumas 2.8 API)

## What this skill does

Applies the six-step in-silico clinical trial workflow from Cortés-Ríos et al.,
*CPT: Pharmacometrics & Systems Pharmacology* (2025),
DOI [10.1002/psp4.70122](https://doi.org/10.1002/psp4.70122), to a Pumas 2.8 NLME model.
This is a **loop, not a pipeline** — every later step can invalidate an earlier choice,
and the user's real work is often in returning to an earlier step with better information.
The router's job is to orient: pick the right entry step, name the back-edges,
and hand off to the step-specific skill without trying to do the step-specific work itself.

## The loop is the important part

ISCT looks linear when you read the tutorials in order (steps 1 → 6),
but in practice users spend most of their time on the back-edges.
Read these carefully before dispatching to a step skill:

- **Step 2 ranking → step 1 reparameterization.**
  A dominant parameter (Sᵢ near 1 for one parameter,
  near 0 for others) suggests a redundant model parameter that can be fixed or removed at step 1.
- **Step 3 non-identifiable → step 1 reparameterization.**
  A parameter that can't be estimated under any realistic measurement scenario needs to be fixed,
  combined with another parameter into an identifiable function,
  or exposed through an additional biomarker (which itself changes step 1's `@observed` /
  `@derived`).
- **Step 3 non-rational RHS → step 1 auxiliary-state reformulation.**
  `StructuralIdentifiability.jl` requires rational ODE right-hand sides.
  Non-integer Hill exponents, fractional powers, and `ifelse` switches break it.
  See `../isct-step3-identifiability/references/non-rational-reformulation.md`.
- **Step 5 INFEASIBLE → step 4 enlarge vpop, or step 1 revisit assumptions.**
  Infeasibility in JuMP+HiGHS calibration is usually a vpop-too-narrow problem first,
  a model-too-rigid problem second.
  Follow the step-5 diagnosis ladder.
- **Step 5 selected vpop too small → step 4 enlarge or step 5 loosen ε.**
  A calibrated vpop with too few patients in a category gives useless step-6 CIs.
  Either relax the per-category ε or regenerate more vpop at step 4.
- **Step 6 CIs too wide / wrong response rate → step 5 or step 4.**
  The diagnosis "selected vpop is too small" usually points backward to calibration tolerance
  or vpop size, not to anything step-6-specific.
- **Step 6 arm comparison looks wrong (identical to control, or diverges implausibly) → step 1
  `@covariates` plumbing.**
  A covariate declared in `@covariates` but not consumed in `@pre` / `@dynamics` is a silent no-op.

## The six steps

| Step | Skill | Julia tool | Artifact produced |
|------|-------|------------|-------------------|
| 1 | [`isct-step1-model`](../isct-step1-model/SKILL.md) | Pumas `@model` | NLME model with `@observed` / `@derived` endpoints, `@covariates` plumbing for arms |
| 2 | [`isct-step2-gsa`](../isct-step2-gsa/SKILL.md) | `GlobalSensitivity.jl` (Sobol, eFAST) | Ranked parameter table — first-order Sᵢ, total-order Sₜᵢ, interaction Sₜᵢ − Sᵢ |
| 3 | [`isct-step3-identifiability`](../isct-step3-identifiability/SKILL.md) | `StructuralIdentifiability.jl` | Per-parameter verdict: globally / locally / non-identifiable; optional identifiable combinations |
| 4 | [`isct-step4-vpop`](../isct-step4-vpop/SKILL.md) | `Copulas.jl` (`GaussianCopula` + `SklarDist`) — or `MvNormal` for all-Normal `@random` | Correlated vpop DataFrame with per-subject η |
| 5 | [`isct-step5-calibration`](../isct-step5-calibration/SKILL.md) | `JuMP` + `HiGHS` (+ `MultiObjectiveAlgorithms` for bi-objective) | Selected vpop subset matching the target outcome distribution |
| 6 | [`isct-step6-vct`](../isct-step6-vct/SKILL.md) | `Pumas.simobs` + `AlgebraOfGraphics` + `HypothesisTests.BinomialTest` | VCT readouts: response-rate CIs, time-course bands, waterfall, spaghetti |

## How to enter

- **User names a specific step** (e.g. "run Sobol sensitivity on my Pumas model") → invoke
  that step skill directly, skip this router.
- **User says "apply ISCT to my model" / "go end-to-end"** → start at step 1 (`isct-step1-model`)
  and move forward, one step at a time.
  Each step skill has an "Outputs to surface for sign-off" section — follow it.
  Do not chain steps silently.
- **User arrives mid-workflow** ("I have a vpop,
  now calibrate it") → jump to the named step (`isct-step5-calibration`),
  but first verify the prerequisite artifacts exist (the step skill lists them).
  If a prerequisite is missing, route back: "You need a step-N artifact;
  let me invoke `isct-stepN` first."
- **Earlier step's output has invalidated a later step** (e.g. calibration is INFEASIBLE) → follow
  the back-edge table above.
  Don't guess — the step-5 skill's loop-back ladder is prescriptive.

## Sign-off between steps

Before each handoff between step skills,
surface the intermediate artifact to the user and get explicit sign-off.
Each step skill has an **Outputs to surface for sign-off** section — treat it as load-bearing.
The concrete artifacts per step:

- Step 1 → typical-patient simulation (via `center_randeffs`), population simulation,
  clinical-validation plot, arm-effect overlay.
- Step 2 → ranked sensitivity DataFrame, stacked first-order + interaction bar chart,
  plain-language ranking summary.
- Step 3 → identifiability-by-scenario table, plain-language verdict per parameter,
  measurement-plan recommendation.
- Step 4 → observed-vs-expected Spearman correlation delta, marginals comparison, corner plot,
  independent-sampling overlay.
- Step 5 → before/after response-distribution table and chart, N and ε_max summary,
  Pareto front (if bi-objective).
- Step 6 → response-rate CI table, time-course band plot, spaghetti, waterfall,
  calibrated-vs-uncalibrated panel.

PumasAide-as-driver: do not silently proceed to the next step.
Each sign-off prompt is the user's chance to loop back with better information.

## When the user's model deviates from the tutorials

The skills cover the common patterns in `tutorials/tb_*` (simple tumor burden)
and `tutorials/hbv_*` (complex QSP).
If the user's model uses constructs the tutorials don't (delay differential equations,
stochastic jumps, a disease area unlike tumor growth or viral dynamics)
and the skill content doesn't map cleanly:

- Read the most relevant tutorial directly before improvising —
  the HBV series makes tribal knowledge explicit that the TB series elides.
- Keep the ISCT step structure;
  the back-edges are still the same even when the specific Julia APIs shift.
- Flag to the user when you're extrapolating beyond the tutorial patterns.

## Tutorial map

- **Tumor burden (simple, 3 parameters)** — `tutorials/tb_0{1..6}_*.qmd`.
  Good for learning the API shape; hides some gotchas.
- **HBV (QSP, 9 estimated + 22 fixed parameters)** — `tutorials/hbv_0{1..6}_*.qmd`.
  Realistic complexity; makes the tribal knowledge explicit (non-rational reformulation,
  resample-until-valid, Everest-trial calibration target, multi-arm time-varying covariates).

When in doubt, read the HBV tutorial for the step you're on.

## Paper citation

> Cortés-Ríos, J. et al. (2025).
> *A Step-by-Step Workflow for Performing In Silico Clinical Trials With Nonlinear Mixed Effects
> Models.*
> CPT: Pharmacometrics & Systems Pharmacology.
> [10.1002/psp4.70122](https://doi.org/10.1002/psp4.70122).

The skill family encodes this workflow in Pumas 2.8.1 specifically;
older Pumas versions have different macro surfaces (`@observed` was required
for GSA endpoints prior to 2.8.1; `@derived` outputs are sufficient in 2.8.1+).
