# GaussianCouplingMessagePassingRules

The `GaussianCoupling` node, the bilinear potential `φ(out, in, a) = exp(out ⋅ a ⋅ in)`, and its
message passing rules. It is the edge potential of Gaussian belief propagation: with
`NormalWeightedMeanPrecision(b[i], A[i, i])` priors and `a = -A[i, j]`, message passing solves
`A x = b`. Its messages are improper normals, and its rules take the structured factorisation
`q(out, in) q(a)` with a constant `a`.

```julia
using GaussianCouplingMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

result = @call_message_update_rule(
    node = GaussianCoupling, target = :in,
    m = (out = NormalMeanVariance(2.0, 3.0),), q = (a = PointMass(-0.5),),
)
getresult(result)   # NormalWeightedMeanPrecision(-1.0, -0.75)
```

- Documentation: `make docs-gaussian-coupling` from the repository root builds it into
  `docs/build`; it will be published at <https://reactivebayes.github.io/GaussianCouplingMessagePassingRules.jl/dev/>.
- Tests: `make test-gaussian-coupling`.
- Depends on MessagePassingRulesBase, BayesBase, ExponentialFamily and Distributions. Julia 1.11
  or later. MIT licence.
