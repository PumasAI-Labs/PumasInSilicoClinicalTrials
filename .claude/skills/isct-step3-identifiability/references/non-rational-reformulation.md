# Auxiliary-state reformulation for non-rational ODE RHS

`StructuralIdentifiability.jl` requires every right-hand side to be a quotient of polynomials in the
state variables and parameters — i.e. **rational**.
Three common pharmacometric constructs violate this:

1. **Hill functions with non-integer exponents** — e.g. `(V + S)^nS` with `nS = 0.486`.
2. **Fractional powers of states** — anything of the form `x(t)^α` with `α ∉ ℤ`.
3. **Threshold switches / `ifelse`** —
   piecewise-defined RHS (e.g. effective-concentration terms active only above a threshold).

This sidecar shows how to rewrite case 1 as a rational system by introducing an auxiliary state;
the same chain-rule trick extends to case 2 unchanged,
and case 3 needs a separate treatment (usually:
drop the switch for the SI analysis and note the restriction).

## The HBV worked example

The HBV QSP model (see `tutorials/hbv_01_model_introduction_tutorial.qmd`) contains the Hill term

```julia
X' = r_X * (1 - X) * (I / (ϕ_E + I)) *
    (1 - S_max * (HBsAg_ug_mL^nS / (ϕ_S^nS + HBsAg_ug_mL^nS))) - d_X * X
```

where

- `HBsAg_ug_mL = c * (V + S)`, a linear function of states `V` and `S`,
- `c = 96 · 24000 · 1e6 / 6.022e23` is a known unit-conversion constant,
- `nS = 0.486` is a non-integer exponent.

The problematic subterm is

$$ \frac{\bigl(c \cdot (V + S)\bigr)^{n_S}}{\phi_S^{n_S} + \bigl(c \cdot (V + S)\bigr)^{n_S}}, $$

because `(V + S)^0.486` is not polynomial in the states.

## The chain-rule trick

Introduce an auxiliary state

$$ H(t) \,\triangleq\, \bigl(c \cdot (V(t) + S(t))\bigr)^{n_S}. $$

By the chain rule,

$$ H'(t) = n_S \bigl(c \cdot (V(t) + S(t))\bigr)^{n_S - 1} \cdot c \cdot \bigl(V'(t) + S'(t)\bigr) = n_S \cdot H(t) \cdot \frac{V'(t) + S'(t)}{V(t) + S(t)}. $$

Now substitute the existing ODEs for `V` and `S`:

- `V'(t) = p_V · ε_NA · ε_IFN · I(t) − d_V · V(t)`
- `S'(t) = p_S · I(t) − d_V · S(t)`

so that

$$ V'(t) + S'(t) = \bigl(p_V \cdot \varepsilon_{NA} \cdot \varepsilon_{IFN} + p_S\bigr) \cdot I(t) - d_V \cdot \bigl(V(t) + S(t)\bigr), $$

giving

$$ H'(t) = n_S \cdot H(t) \cdot \frac{(p_V \cdot \varepsilon_{NA} \cdot \varepsilon_{IFN} + p_S) \cdot I(t) - d_V \cdot (V(t) + S(t))}{V(t) + S(t)}. $$

This right-hand side is **rational** in `H`, `I`, `V`, `S` and the parameters —
quotient of polynomials, no fractional powers.

The Hill term in `X'` becomes

$$ \frac{H(t)}{\psi + H(t)}, \qquad \psi \,\triangleq\, \phi_S^{n_S}, $$

where `ψ` is a known constant (fixed parameter), also rational.

## Initial condition for the auxiliary state

`H` must start at the value implied by its definition at t=0:

$$ H(0) = \bigl(c \cdot (V(0) + S(0))\bigr)^{n_S}. $$

In Pumas, add this to `@init`:

```julia
@init begin
    # … existing initial conditions for V, S, I, …
    H = (c * (ini_V + p_S * ini_V / p_V))^nS     # V(0) + S(0) evaluated at steady state
end
```

Use whatever analytic or steady-state form your model uses for the original `V`
and `S` initial conditions; the auxiliary state piggybacks on them.

## The reformulated `@dynamics`

```julia
@dynamics begin
    # … existing ODEs (unchanged except for X') …
    X' = r_X * (1 - X) * (I / (ϕ_E + I)) *
        (1 - S_max * (H / (ψ + H))) - d_X * X       # Hill term now rational
    H' = nS * H * ((p_V * ϵ_NA * ϵ_IFN + p_S) * I - d_V * (V + S)) / (V + S)
end
```

Now `StructuralIdentifiability.jl` can consume `model.sys` directly.
See `tutorials/hbv_03_structural_identifiability_tutorial.qmd` lines 119–286
for the full reformulated model.

## General recipe

For any non-rational term `f(x(t))` where `f` is smooth but non-rational (fractional power,
logarithm, exponential of a state):

1. Introduce `H(t) = f(x(t))` as a new state.
2. Compute `H'(t) = f'(x(t)) · x'(t)` by the chain rule,
   then substitute `x'(t)` from the existing dynamics.
3. Simplify until the right-hand side is rational in the existing states, `H`, and the parameters.
4. Add `H'` to `@dynamics` and `H(0)` to `@init`.
5. Replace every occurrence of `f(x(t))` in the other equations with `H(t)`.

If step 3 doesn't yield a rational form, the transformation won't help —
consider whether the offending term can be linearized (for SI purposes only) by a Padé-style
rational approximation, or whether you need a different tool (practical identifiability via profile
likelihoods, for example).

## Limitations

- The reformulation changes the **model's algebraic skeleton** but is chosen so
  that simulation behaviour is preserved exactly (on the invariant set `H = f(x)`).
  Use the reformulated model for SI analysis; the original model for simulation and fitting.
- Conditionals (`ifelse`, threshold switches) cannot be reformulated this way.
  For SI, drop the switch and analyse the dominant regime;
  document the restriction in the identifiability conclusion.
- The auxiliary state adds to the system's dimension —
  local identifiability screens will be slightly slower, global analysis noticeably so.
  Usually worth it; verify by running local-first.

## When to use this sidecar

Load this file when:

- The model has a Hill function with a non-integer exponent.
- Any state appears under a fractional power.
- `StructuralIdentifiability.jl` errors with a "not rational" complaint.
- You're reviewing step 1 for ISCT-readiness and spot a non-rational term —
  reformulate preemptively to avoid a step-3 rework.
