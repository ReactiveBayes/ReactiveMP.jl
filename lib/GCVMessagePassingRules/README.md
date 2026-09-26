# GCVMessagePassingRules

The Gaussian controlled variance node, `GCV`, for reactive message passing:
`y ~ N(x, exp(κz + ω))`, a normal whose log-variance is linear in other variables, the building
block of hierarchical Gaussian filters. The package holds the node, its algorithm
`GCVApproximation`, the density `ExponentialLinearQuadratic` of its messages towards `z`, `κ` and
`ω`, and the rules that let `NormalMeanVariance` and `NormalMeanPrecision` take such a message on
their `out`. The rules are variational, under `q(y, x) q(z) q(κ) q(ω)` or the mean field.

```julia
using MessagePassingRulesBase, GCVMessagePassingRules, ExponentialFamily, BayesBase

result = @call_message_update_rule(
    node = GCV, target = :y,
    m = (x = NormalMeanVariance(1.0, 2.0),),
    q = (z = PointMass(0.0), κ = PointMass(1.0), ω = PointMass(0.0)),
)
getresult(result)   # NormalMeanVariance(1.0, 3.0): the noise variance exp(1 ⋅ 0 + 0) = 1 added
```

- Documentation: `make docs-gcv` from the repository root builds it into `docs/build`; it will be
  published at <https://reactivebayes.github.io/GCVMessagePassingRules.jl/dev/>.
- Tests: `make test-gcv`.
- Depends on MessagePassingRulesBase, MessagePassingRulesApproximations (the Gauss–Hermite
  cubature), StandardMessagePassingRules (the normal nodes it extends), BayesBase,
  ExponentialFamily, Distributions and StatsFuns. Julia 1.11 or later. MIT licence.
