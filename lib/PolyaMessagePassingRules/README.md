# PolyaMessagePassingRules

**This package is licensed under GPL-3**, because its dependency PolyaGammaHybridSamplers is.
The engine and the other rule packages are MIT and do not depend on it.

The Pólya-Gamma augmented nodes for count data through the logistic, whose messages towards the
weights are normal:

- `BinomialPolya`, `y ~ Binomial(n, σ(xᵀβ))`: binomial and logistic regression, under
  `BinomialPolyaApproximation(; samples = nothing)`;
- `MultinomialPolya`, `x ~ Multinomial(N, p(ψ))` by logistic stick-breaking: multinomial
  regression, under `MultinomialPolyaApproximation(; points = 21)`.

Both algorithms are the nodes' defaults. The rules towards the weights read the message on the
weights' own edge, which a model must initialise.

```julia
using PolyaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily

result = @call_message_update_rule(
    node = BinomialPolya, target = :β,
    q = (y = PointMass(3), x = PointMass([0.1, 0.2]), n = PointMass(5)),
    m = (β = MvNormalWeightedMeanPrecision(zeros(2), [1.0 0.0; 0.0 1.0]),),
)
getresult(result)   # MvNormalWeightedMeanPrecision, ξ = (3 - 5/2) x, Λ = (5/4) x xᵀ
```

- Documentation: `make docs-polya` from the repository root builds it into `docs/build`; it
  will be published at <https://reactivebayes.github.io/PolyaMessagePassingRules.jl/dev/>.
- Tests: `make test-polya`.
- Depends on MessagePassingRulesBase, MessagePassingRulesApproximations (Gauss–Hermite
  cubature), BayesBase, ExponentialFamily, Distributions, LogExpFunctions, SpecialFunctions and
  PolyaGammaHybridSamplers.
- Licence: **GPL-3**.
