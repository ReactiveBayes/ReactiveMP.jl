# SoftDotMessagePassingRules

The SoftDot node, `y ~ N(θᵀx, γ⁻¹)`: a dot product of `θ` and `x` softened by Gaussian noise of
precision `γ`, with its variational rules, all in closed form. It serves Bayesian linear
regression with a learnt noise precision, and dot products of two Gaussian unknowns, which the
deterministic dot product node has no closed form for. The rules run under mean-field or the
structured `q(y, x)q(θ)q(γ)`, with an average energy for each; there are no belief-propagation
rules.

```julia
using MessagePassingRulesBase, SoftDotMessagePassingRules, ExponentialFamily

q = (y = NormalMeanVariance(1.0, 1.0), θ = NormalMeanVariance(1.0, 1.0), x = NormalMeanVariance(1.0, 1.0))
getresult(@call_message_update_rule(node = SoftDot, target = :γ, q = q))   # GammaShapeRate(1.5, 2.0)
```

- Documentation: `make docs-softdot` from the repository root builds it into `docs/build`; it
  will be published at <https://reactivebayes.github.io/SoftDotMessagePassingRules.jl/dev/>.
- Tests: `make test-softdot`.
- Depends on MessagePassingRulesBase, StandardMessagePassingRules, BayesBase, ExponentialFamily,
  Distributions and StatsFuns; not on the autoregressive package. Julia 1.11 or later. MIT
  licence.
