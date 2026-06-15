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

"""
    conjugatear_effective_marginals(q_w::MvNormalGamma)

Map the joint marginal `q(θ, γ) = MvNormalGamma(μ, Λ, α, β)` to the `(q_θ, q_γ)` pair under
which the AR state/energy rules reproduce the exact ConjugateAR moments: `E[γ] = α/β`,
`E[θ] = μ`, and the coefficient–precision coupling `mγ·Vθ = Λ⁻¹` (i.e. `E[γθθᵀ] − E[γ]μμᵀ`).
This lets the `ConjugateAR` state rules delegate to the tested `AR` rules without duplicating
their algebra.
"""
function conjugatear_effective_marginals(q_w::MvNormalGamma)
    μ, Λ, α, β = params(q_w)
    mγ = α / β
    q_θ = MvNormalMeanCovariance(μ, cholinv(Λ) / mγ)
    q_γ = GammaShapeRate(α, β)
    return q_θ, q_γ
end

@average_energy ConjugateAR (
    q_y_x::MultivariateNormalDistributionsFamily, q_w::MvNormalGamma, meta::ARMeta
) = begin
    q_θ, q_γ = conjugatear_effective_marginals(q_w)
    return score(
        AverageEnergy(),
        AR,
        Val{(:y_x, :θ, :γ)}(),
        (Marginal(q_y_x, false, false), Marginal(q_θ, false, false), Marginal(q_γ, false, false)),
        meta,
    )
end
