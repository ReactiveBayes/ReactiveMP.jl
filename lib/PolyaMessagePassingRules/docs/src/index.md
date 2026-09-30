# PolyaMessagePassingRules

!!! warning "Licence: GPL-3"
    This package is licensed under **GPL-3**, because its dependency PolyaGammaHybridSamplers is.
    The engine and the other rule packages are MIT and do not depend on it; a model that loads
    this package is subject to the GPL.

Two [factor nodes](@extref MessagePassingRulesBase glossary-factor-node) for count data through
the logistic, [`BinomialPolya`](@ref) and [`MultinomialPolya`](@ref). Both are augmented with
Pólya-Gamma variables, so that the [messages](@extref MessagePassingRulesBase glossary-message)
towards their weights are normal and a normal prior on the weights stays conjugate. Use them for
binomial or logistic regression, and for multinomial regression.

```@docs
PolyaMessagePassingRules
```

!!! info "Where these rules run"
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs these rules on a factor
    graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model. The examples here call the
    rules directly, as a test does.

## [The augmentation](@id polya-augmentation)

For a count `y` out of `b` trials with log-odds `ψ`, the Pólya-Gamma identity of Polson, Scott
and Windle (2013),

```math
\frac{(e^{\psi})^{y}}{(1 + e^{\psi})^{b}}
  = 2^{-b} e^{\kappa \psi} \int_0^\infty e^{-\omega \psi^2 / 2} \, \mathrm{PG}(\omega \mid b, 0) \, d\omega,
\qquad \kappa = y - b / 2,
```

makes the likelihood Gaussian in `ψ` given `ω ~ PG(b, ψ)`. The
[rules](@extref MessagePassingRulesBase glossary-rule) replace `ω` by its mean at the current
estimate of `ψ`, read from the message on the weights' own edge. A model must therefore give that
message an [initial message](@extref MessagePassingRulesBase glossary-initial-message). The
[average energies](@extref MessagePassingRulesBase glossary-average-energy) are those of the
original likelihoods, not of the augmented ones.

## Pages

- [BinomialPolya](@ref page-binomial-polya): `y ~ Binomial(n, σ(xᵀβ))`, with its algorithm
  [`BinomialPolyaApproximation`](@ref).
- [MultinomialPolya](@ref page-multinomial-polya): `x ~ Multinomial(N, p(ψ))` by logistic
  stick-breaking, with its algorithm [`MultinomialPolyaApproximation`](@ref) and the helpers
  [`logistic_stick_breaking`](@ref) and [`compose_Nks`](@ref).
