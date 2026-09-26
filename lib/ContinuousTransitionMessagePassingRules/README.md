# ContinuousTransitionMessagePassingRules

The `ContinuousTransition` node, `y ~ N(f(a) x, W⁻¹)` (alias `CTransition`): a linear Gaussian
transition from `x` to `y` through a matrix built from the vector `a`, with `a` and the noise
precision `W` learned along with the states. Its rules are variational, mean-field or structured
`q(y, x)`, with average energies, and run under `CTVMP(f)`, which carries the transformation `f`
and which a model must give. A nonlinear `f` is linearised with ForwardDiff.

```julia
using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily

result = @call_message_update_rule(
    node = ContinuousTransition, target = :y, algorithm = CTVMP(a -> reshape(a, 1, 1)),
    q = (x = MvNormalMeanCovariance([2.0], [1.0;;]), a = MvNormalMeanCovariance([3.0], [1.0;;]), W = Wishart(2, [0.5;;])),
)
getresult(result)   # MvNormalMeanPrecision([6.0], [1.0;;])
```

- Documentation: `make docs-continuous-transition` from the repository root builds it into
  `docs/build`; it will be published at
  <https://reactivebayes.github.io/ContinuousTransitionMessagePassingRules.jl/dev/>.
- Tests: `make test-continuous-transition`.
- Depends on MessagePassingRulesBase, StandardMessagePassingRules, BayesBase, ExponentialFamily,
  Distributions, FastCholesky and ForwardDiff.
- MIT licence.
