# Design: `MvNormalGamma` conjugate update for the Autoregressive node

**Branch:** `add-ar-mvnormalgamma-rule`
**Status:** Derivation + spec (no implementation yet — review gate before coding)
**Depends on:** `ExponentialFamily.jl#mv-normal-gamma` (PR #287, `MvNormalGamma` distribution + `prod` rule)

## 1. Motivation

The `AR` node (`src/nodes/predefined/autoregressive.jl`, edges `[y, x, θ, γ]`) models

```math
y_{1} \mid x, \theta, \gamma \;\sim\; \mathcal N(\theta^\top x,\; \gamma^{-1}),
```

where `y₁` is the first component of the next state, `x` the lagged state, `θ` the AR
coefficients and `γ` the scalar transition precision.

Today the coefficients and the precision are updated under a **mean-field** split
`q(θ, γ) = q(θ) q(γ)`:

- `AR(:θ, Marginalisation)` → `MvNormal` (`src/rules/autoregressive/theta.jl`)
- `AR(:γ, Marginalisation)` → `GammaShapeRate` (`src/rules/autoregressive/gamma.jl`)

The mean-field split ignores the posterior correlation between `θ` and `γ`. Because
`(θ, γ)` are *jointly conjugate* for this likelihood (it is a Bayesian linear regression
in `θ` with unknown noise precision `γ`), the exact structured posterior is a
**Normal-Gamma**. Keeping `q(θ, γ)` joint as an `MvNormalGamma` is therefore both more
accurate and closed-form.

This document derives the required variational message-passing (VMP) rule and specifies
the implementation. The two reference results it rests on:

- Normal-Gamma density — <https://statproofbook.github.io/P/ng-pdf>
- Bayesian-linear-regression Normal-Gamma posterior — <https://statproofbook.github.io/P/blr-post>

## 2. Distributions

### 2.1 Normal-Gamma (`ng-pdf`)

`NG(μ, Λ, a, b)` over `(x ∈ ℝⁿ, τ > 0)`:

```math
x \mid \tau \sim \mathcal N\!\big(\mu, (\tau\Lambda)^{-1}\big), \qquad \tau \sim \mathrm{Gam}(a, b),
```

```math
p(x,\tau) = \sqrt{\frac{|\Lambda|}{(2\pi)^n}}\,\frac{b^a}{\Gamma(a)}\;
            \tau^{\,a+\frac n2-1}\,
            \exp\!\Big[-\tfrac{\tau}{2}\big((x-\mu)^\top\Lambda(x-\mu)+2b\big)\Big].
```

This is `ExponentialFamily.MvNormalGamma(μ, Λ, α, β)` (mean parametrization
`(μ, Λ, α, β)`); `n = 1` recovers `ExponentialFamily.NormalGamma`. The branch already
provides `params`, `mean`, `logpdf`, the natural-parameter maps, and `prod`.

### 2.2 Sufficient statistics / expected moments

From `getsufficientstatistics(MvNormalGamma)` and `getgradlogpartition`, with
`(μ, Λ, α, β)`:

```math
T(\theta,\gamma) = \big(\gamma\theta,\; \gamma\theta\theta^\top,\; \log\gamma,\; \gamma\big),
```

```math
\mathbb E[\gamma] = \tfrac{\alpha}{\beta},\quad
\mathbb E[\gamma\theta] = \tfrac{\alpha}{\beta}\mu,\quad
\mathbb E[\gamma\theta\theta^\top] = \Lambda^{-1} + \tfrac{\alpha}{\beta}\mu\mu^\top,\quad
\mathbb E[\log\gamma] = \psi(\alpha) - \log\beta.
```

These are exactly the moments the outbound `x`/`y` messages and the average-energy need
(Section 5).

## 3. The conjugate update (marginal rule)

### 3.1 Likelihood as a function of `(θ, γ)`

Under the structured factorization `q(y, x, θ, γ) = q(y, x)\, q(θ, γ)`, the joint
marginal is

```math
q(\theta,\gamma) \;\propto\; \overrightarrow{m}_{\theta\gamma}(\theta,\gamma)\;
        \exp\!\big(\mathbb E_{q(y,x)}[\log f(y,x,\theta,\gamma)]\big),
```

where `m_{θγ}` is the inbound message on the joint `(θ, γ)` edge (the prior) and `f` is
the AR factor. Taking the expectation over `q(y,x)` only (we are *not* averaging over
`θ`):

```math
\mathbb E_{q(y,x)}[\log f]
  = \tfrac12\log\gamma - \tfrac{\gamma}{2}\,
    \mathbb E_{q(y,x)}\!\big[(y_1-\theta^\top x)^2\big] + \text{const},
```

```math
\mathbb E_{q(y,x)}[(y_1-\theta^\top x)^2]
  = \underbrace{\mathbb E[y_1^2]}_{a}
  - 2\,\theta^\top\underbrace{\mathbb E[x\,y_1]}_{b}
  + \theta^\top\underbrace{\mathbb E[xx^\top]}_{C}\theta .
```

So the likelihood contribution is itself Normal-Gamma-shaped, with **natural
parameters**

```math
\eta^{\text{lik}} = \big(\,b,\; -\tfrac12 C,\; \tfrac12,\; -\tfrac12 a\,\big)
```

against the statistic order `(γθ, γθθᵀ, log γ, γ)`.

### 3.2 Expected sufficient statistics from `q(y, x)`

`q(y,x)` is `MultivariateNormalDistributionsFamily`; slice it exactly as the existing
`theta.jl` / `gamma.jl` do (`F = getvform(meta)`, `order = getorder(meta)`):

```julia
myx, Vyx_full = mean_cov(q_y_x)
my  = ar_slice(F, myx, 1:order)
mx  = ar_slice(F, myx, (order+1):(2order))
Vy  = ar_slice(F, Vyx_full, 1:order, 1:order)
Vx  = ar_slice(F, Vyx_full, (order+1):(2order), (order+1):(2order))
Vxy = ar_slice(F, Vyx_full, (order+1):(2order), 1:order)   # rows = x, cols = y
my1, Vy1 = first(my), first(Vy)
```

Then (with `c = ar_unit(T, F, order) = e₁`):

```math
C = V_x + m_x m_x^\top \quad(=\mathbb E[xx^\top]),\qquad
b = V_{xy}\,e_1 + m_x\,m_{y_1} \quad(=\mathbb E[x\,y_1]),\qquad
a = V_{y_1} + m_{y_1}^2 \quad(=\mathbb E[y_1^2]).
```

`C` is `rank1update(Vx, mx)`; `b` is `rank1update(Vxy, mx, my) * c` — both already used in
`theta.jl`. (Univariate: every quantity is scalar, `Vxy` is `cov(x, y)`.)

### 3.3 Posterior hyperparameters (BLR result, `blr-post`)

Map the Bayesian-linear-regression posterior (precision form, `P = 1`, `n = 1`) onto the
AR quantities — `Xᵀ P X → C`, `Xᵀ P y → b`, `yᵀ P y → a`:

Given prior `m_{θγ} = NG(μ₀, Λ₀, α₀, β₀)`:

```math
\boxed{
\begin{aligned}
\Lambda_n &= \Lambda_0 + C, \\
\mu_n     &= \Lambda_n^{-1}\,(\Lambda_0\mu_0 + b), \\
\alpha_n  &= \alpha_0 + \tfrac12, \\
\beta_n   &= \beta_0 + \tfrac12\big(a + \mu_0^\top\Lambda_0\mu_0 - \mu_n^\top\Lambda_n\mu_n\big).
\end{aligned}}
```

Return `MvNormalGamma(μn, Λn, αn, βn)`.

### 3.4 Correctness anchors

1. **Equivalence to `prod`.** Writing the Section-3.1 likelihood as an `MvNormalGamma`
   and multiplying via the branch's
   `prod(::PreserveTypeProd, ::MvNormalGamma, ::MvNormalGamma)` reproduces the boxed
   equations **algebraically** (verified by hand: the `prod`'s
   `α = αˡ+αʳ+d/2-1` and `β = …` collapse to `α₀+½` and the boxed `βₙ`). We implement the
   boxed update directly rather than constructing the (possibly improper, `αˡ = 3/2−d/2`)
   likelihood factor as a distribution.
2. **Univariate reduction.** For `order = 1`, `MvNormalGamma → NormalGamma` and the update
   equals the scalar Normal-Gamma AR update.
3. **Consistency with mean-field.** Marginalizing the joint `q(θ,γ)` should give a `q(θ)`
   (a multivariate Student-t; its mean is `μn`) and `q(γ) = Gam(αn, βn)` consistent in
   the large-data limit with iterating the separate `theta.jl`/`gamma.jl` rules.

## 4. Primary rule — `@marginalrule AR(:θ_γ)`

New file `src/rules/autoregressive/theta_gamma.jl`:

```julia
@marginalrule AR(:θ_γ) (
    m_θ_γ::MvNormalGamma,
    q_y_x::MultivariateNormalDistributionsFamily,
    meta::ARMeta,
) = begin
    # slice q(y,x) → C, b, a   (Section 3.2)
    # BLR update → (μn, Λn, αn, βn)   (Section 3.3)
    return MvNormalGamma(μn, Λn, αn, βn)
end
```

Notes:
- The inbound `m_θ_γ` is the `MvNormalGamma` prior message; in a temporal/shared-prior
  graph this is the message from the previous slice / shared coefficient prior.
- `ARsafe` regularization: reuse the node's existing safeguards when forming `Λn`/its
  inverse (`cholinv`), mirroring the `ARsafe`/`ARunsafe` handling in `marginals.jl`.

## 5. Supporting rules (needed for end-to-end use of the joint marginal)

These let the rest of the graph consume `q_θ_γ::MvNormalGamma`. They mirror the existing
mean-field rules but substitute the *joint* moments from Section 2.2 for the independent
products `mγ·mθ`, `mγ·(Vθ+mθmθᵀ)`:

| Rule | File | Replaces / adds |
|------|------|-----------------|
| `AR(:x, Marginalisation) (m_y, q_θ_γ::MvNormalGamma, meta)` | `x.jl` | uses `E[γ]`, `E[γθ]`, `E[γθθᵀ]` |
| `AR(:y, Marginalisation) (m_x, q_θ_γ::MvNormalGamma, meta)` | `y.jl` | same |
| `@average_energy AR (q_y_x, q_θ_γ::MvNormalGamma, meta)` | `autoregressive.jl` | `E[logγ]=ψ(α)−logβ`, `E[γθθᵀ]` |

Substitution map (from the existing `q_θ` + `q_γ` forms to the joint form):

```math
m_\gamma \to \tfrac{\alpha}{\beta},\qquad
m_\gamma m_\theta \to \tfrac{\alpha}{\beta}\mu = \mathbb E[\gamma\theta],\qquad
m_\gamma(V_\theta + m_\theta m_\theta^\top) \to \Lambda^{-1} + \tfrac{\alpha}{\beta}\mu\mu^\top = \mathbb E[\gamma\theta\theta^\top].
```

These are mechanical once the marginal rule and its tests are in place; they carry the
small risk of re-deriving the `add_transition`/companion-matrix algebra, so they land in a
second commit after the marginal rule is green.

### 5.1 Outbound message `AR(:θ_γ)` (deferred)

The standalone outbound message toward the prior is the Section-3.1 likelihood, whose
`MvNormalGamma` form can be improper (`α = 3/2 − d/2 < 1` for `d ≥ 2`). It is only needed
for temporal coupling of `(θ,γ)` across time and is **out of scope** for this branch; if
required later it must be carried in natural-parameter form
(`ExponentialFamilyDistribution`), not as an `MvNormalGamma` struct.

## 6. Registration & wiring

- Add `include("autoregressive/theta_gamma.jl")` to `src/rules/predefined.jl`
  (after `marginals.jl`, line ~128).
- No new export needed (`MvNormalGamma` comes from `ExponentialFamily`, already a dep and
  re-exported).
- GraphPPL model-level usage — placing a joint `(θ, γ) ~ MvNormalGamma(...)` prior and the
  `q(y,x)q(θ,γ)` factorization on the AR node — is the **end-to-end** step and is *not*
  part of this doc’s scope; it is gated behind the rules + tests landing first.

## 7. Test matrix (TDD, `test/rules/autoregressive/`)

`theta_gamma_tests.jl` (new):

1. **Natural-parameter form.** For a fixed `q_y_x` and unit prior, assert the returned
   `(μn, Λn, αn, βn)` equal the boxed formulas computed independently from `C, b, a`.
2. **`prod` equivalence.** Construct the likelihood `MvNormalGamma` from `η^lik` and assert
   `marginalrule` output ≈ `prod(prior, likelihood)` (where the likelihood is proper, e.g.
   `order = 1`).
3. **Univariate ↔ `NormalGamma`.** `order = 1` result matches a hand-coded scalar
   Normal-Gamma update and `ExponentialFamily.NormalGamma`.
4. **`α` bookkeeping.** `αn == α₀ + 1/2` exactly, independent of data.
5. **PSD / properness.** `Λn` positive-definite and `βn > 0` for valid inputs; `ARsafe`
   path stays finite for near-singular `C`.
6. **Recovery sanity.** With a tight, informative `q_y_x` centered on known
   `(θ*, γ*)`, `μn ≈ θ*` and `αn/βn ≈ γ*`.
7. **Type/variate promotion.** `Float64`/`BigFloat`, `Univariate`/`Multivariate` orders
   `1,2,3` via `@test_rules` harness (match style of existing AR rule tests).

Supporting-rule tests (second commit): extend `x.jl`/`y.jl`/average-energy test files with
`q_θ_γ::MvNormalGamma` cases, cross-checking against the mean-field rules when the joint
collapses to independence (`Λ` diagonal-dominant, `α` large).

## 8. Open questions for review

1. **Edge grouping.** Confirm the intended factorization is `q(y,x) q(θ,γ)` on the
   existing 4-edge node (Section 1), vs. introducing a combined `(θ,γ)` interface.
2. **Scope of supporting rules.** Land only the marginal rule + tests now, or include the
   `x`/`y`/average-energy joint forms in the same PR (Section 5)?
3. **End-to-end model.** Whether/when to add the GraphPPL joint-prior model + a free-energy
   recovery test (mirrors the `scratch/ar_inference_test.jl` smoke test on the RxInfer
   side).
