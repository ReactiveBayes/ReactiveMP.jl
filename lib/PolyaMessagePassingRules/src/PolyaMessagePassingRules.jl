"""
    PolyaMessagePassingRules

The Pólya-Gamma augmented nodes, for count data through the logistic:

- [`BinomialPolya`](@ref), `y ~ Binomial(n, σ(xᵀβ))`, a binomial (or logistic) regression with
  covariates `x` and weights `β`, under [`BinomialPolyaApproximation`](@ref);
- [`MultinomialPolya`](@ref), `x ~ Multinomial(N, p(ψ))` through logistic stick-breaking, under
  [`MultinomialPolyaApproximation`](@ref), with the helpers [`logistic_stick_breaking`](@ref) and
  [`compose_Nks`](@ref).

Pólya-Gamma augmentation makes the message towards the weights normal, so a normal prior on them
stays conjugate. Each rule towards the weights reads the message on its own edge, which a model
must initialise. Both nodes have average energies.

**This package is licensed under GPL-3**, through its dependency PolyaGammaHybridSamplers, which
is. The engine and the other rule packages do not depend on it.

# Examples

```jldoctest; setup = :(using PolyaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> result = @call_message_update_rule(
           node = BinomialPolya, target = :β,
           q = (y = PointMass(3), x = PointMass(0.1), n = PointMass(5)),
           m = (β = NormalWeightedMeanPrecision(0.0, 1.0),),
       );

julia> weightedmean(getresult(result)) ≈ 0.05 && precision(getresult(result)) ≈ 0.0125
true
```

The weighted mean is `(y - n / 2) x` and the precision `ω x²`, `ω = n / 4` being the Pólya-Gamma
mean at `xᵀβ = 0`.
"""
module PolyaMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using MessagePassingRulesApproximations: ghcubature, getweights, getpoints
using BayesBase: promote_samplefloattype, promote_variate_type
using LinearAlgebra: dot, Diagonal
using LogExpFunctions: logistic, softplus
using SpecialFunctions: loggamma
using PolyaGammaHybridSamplers: PolyaGammaHybridSampler

export BinomialPolya, BinomialPolyaApproximation
export MultinomialPolya, MultinomialPolyaApproximation, logistic_stick_breaking, compose_Nks

# The Pólya-Gamma identity both nodes are augmented by.
const DOC_POLYA_AUGMENTATION = rstrip(
    raw"""
    For a count `y` out of `b` trials with log-odds `ψ`, the Pólya-Gamma identity

    ```math
    \frac{(e^{\psi})^{y}}{(1 + e^{\psi})^{b}}
      = 2^{-b} e^{\kappa \psi} \int_0^\infty e^{-\omega \psi^2 / 2} \, \mathrm{PG}(\omega \mid b, 0) \, d\omega,
    \qquad \kappa = y - b / 2,
    ```

    makes the likelihood Gaussian in `ψ` given the auxiliary `ω ~ PG(b, ψ)`: proportional to
    `exp(κψ - ωψ²/2)`. The rules replace `ω` by its expectation, `E[ω] = b / (2c) · tanh(c / 2)`
    for `PG(b, c)`, at the current estimate `c` of `ψ`, so the message towards the weights is a
    normal in weighted-mean and precision form.
    """
)

include("binomial_polya.jl")
include("multinomial_polya.jl")

end
