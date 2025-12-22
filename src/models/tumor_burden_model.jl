"""
    Tumor Burden Model

Implements the tumor burden model from Qi & Cao (2023) as described in
Section 3.1 of the ISCT workflow paper.

Model Equation (Equation 1 in paper):
    N_t = N_0 * f * e^(-k*t) + N_0 * (1-f) * e^(g*t)

Where:
- N_0 = 1 (baseline-normalized tumor size)
- f = fraction of treatment-sensitive cells (0 to 1)
- k = death rate for sensitive cells (d⁻¹)
- g = growth rate for resistant/insensitive cells (d⁻¹)

Treatment arms:
- Control (treatment=0): No drug effect, tumor grows exponentially
- Treatment (treatment=1): Drug kills sensitive fraction, resistant fraction grows

Reference:
    Qi T, Cao Y. Virtual Clinical Trials: A Tool for Predicting Patients Who May
    Benefit From Treatment Beyond Progression With Pembrolizumab in Non-Small Cell
    Lung Cancer. CPT Pharmacometrics Syst Pharmacol. 2023;12:236-249.
"""

using Pumas

"""
    tumor_burden_model

Pumas model for tumor burden dynamics during chemotherapy.

The model uses an ODE formulation that separates tumor into two populations:
- Treatment-sensitive cells (N_sens): die at rate k when treated
- Treatment-resistant cells (N_res): grow at rate g regardless of treatment

Covariates:
- `treatment`: 0 for control arm, 1 for treatment arm

Parameters with inter-individual variability:
- `f`: Treatment-sensitive fraction (logit-normal, range [0,1])
- `g`: Growth rate (logit-normal, range [0, 0.13] d⁻¹)
- `k`: Death rate (logit-normal, range [0, 1.6] d⁻¹)
"""
const tumor_burden_model = @model begin
    @param begin
        # Population parameters (typical values from Qi & Cao 2023)
        tvf ∈ RealDomain(lower=0.0, upper=1.0, init=0.27)
        tvg ∈ RealDomain(lower=0.0, init=0.0013)
        tvk ∈ RealDomain(lower=0.0, init=0.0091)

        # Inter-individual variability (omega values for logit-normal)
        # These are on the logit-transformed scale
        Ω ∈ PDiagDomain(init=[2.16, 1.57, 1.24])

        # Residual error
        σ ∈ RealDomain(lower=0.0, init=0.05)
    end

    @random begin
        # Random effects (on logit/log scale)
        η ~ MvNormal(Ω)
    end

    @covariates treatment

    @pre begin
        # Transform random effects to individual parameters
        # Using logit-normal transformation as in paper

        # Individual f (treatment-sensitive fraction)
        # Bounded [0, 1]
        f = tvf * exp(η[1]) / (1 + tvf * (exp(η[1]) - 1))

        # Individual g (growth rate)
        # Log-normal approximation for positive values
        g = tvg * exp(η[2])

        # Individual k (death rate)
        # Log-normal approximation for positive values
        k = tvk * exp(η[3])

        # Effective death rate depends on treatment
        # treatment = 0 → no killing (control arm)
        # treatment = 1 → killing at rate k (treatment arm)
        k_eff = treatment * k

        # Initial conditions for the two tumor populations
        # Total initial tumor size is normalized to 1
        N_sens_0 = f           # Sensitive population
        N_res_0 = 1.0 - f      # Resistant population
    end

    @init begin
        N_sens = N_sens_0
        N_res = N_res_0
    end

    @dynamics begin
        # Sensitive cells: die at effective rate (0 if control, k if treatment)
        N_sens' = -k_eff * N_sens

        # Resistant cells: always grow at rate g
        N_res' = g * N_res
    end

    @derived begin
        # Total normalized tumor diameter
        Nt = @. N_sens + N_res

        # Observation with residual error
        tumor_size ~ @. Normal(Nt, σ)
    end

    @observed begin
        # GSA endpoints using NCA
        nca := @nca Nt
        final_tumor = NCA.clast(nca)
        auc_tumor = NCA.auc(nca)
    end
end

"""
    tumor_burden_model_analytical

Alternative implementation using analytical solution directly.
More efficient for simulation, useful for large virtual populations.

This model computes the analytical solution:
    N_t = f * e^(-k*t) + (1-f) * e^(g*t)   for treatment
    N_t = e^(g*t)                            for control
"""
const tumor_burden_model_analytical = @model begin
    @param begin
        tvf ∈ RealDomain(lower=0.0, upper=1.0, init=0.27)
        tvg ∈ RealDomain(lower=0.0, init=0.0013)
        tvk ∈ RealDomain(lower=0.0, init=0.0091)
        Ω ∈ PDiagDomain(init=[2.16, 1.57, 1.24])
        σ ∈ RealDomain(lower=0.0, init=0.05)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates treatment

    @pre begin
        f = tvf * exp(η[1]) / (1 + tvf * (exp(η[1]) - 1))
        g = tvg * exp(η[2])
        k = tvk * exp(η[3])
        k_eff = treatment * k
    end

    @derived begin
        # Analytical solution at observation times
        Nt = @. f * exp(-k_eff * t) + (1 - f) * exp(g * t)
        tumor_size ~ @. Normal(Nt, σ)
    end
end

# Note: tumor_burden_model_fixed has been removed.
# Use zero_randeffs(tumor_burden_model, population, params) for deterministic simulations.
# Or pass individual parameters as random effects directly to simobs().

export tumor_burden_model, tumor_burden_model_analytical
