export ConjugateAR

@doc raw"""
    ConjugateAR

Autoregressive node with a **combined conjugate parameter interface**. The likelihood is
identical to [`AR`](@ref) — `y₁ ∼ N(θᵀx, γ⁻¹)` with companion-form state propagation — but
the AR coefficients `θ` and the transition precision `γ` enter through a *single* edge `w`
that carries the joint `(θ, γ)` as an `MvNormalGamma` variable, instead of the two
mean-field edges `θ` and `γ` of [`AR`](@ref).

Keeping `(θ, γ)` joint makes the parameter update exactly conjugate: under the structured
factorization `q(y, x) q(θ, γ)` the posterior `q(θ, γ)` is `MvNormalGamma`. This is a
Bayesian linear regression with unknown noise precision, where the regression statistics
are the expected sufficient statistics of `q(y, x)`. The node reuses [`ARMeta`](@ref) for
the order and variate form.

# Interfaces
1. `y` (alias `out`) — current state.
2. `x` — lagged state vector of length `order`.
3. `w` — joint parameters `(θ, γ)` distributed as `MvNormalGamma`.

See `design/ar_mvnormalgamma_conjugate_rule.md` for the derivation.
"""
struct ConjugateAR end

@node ConjugateAR Stochastic [(y, aliases = [out]), x, w]

default_meta(::Type{ConjugateAR}) = error(
    "ConjugateAR node requires a meta (ARMeta) flag explicitly specified"
)
