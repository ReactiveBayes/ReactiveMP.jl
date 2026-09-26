# DeltaMessagePassingRules

The Delta node, `out = f(in₁, …, inₙ)` for any deterministic function `f`, and its message
passing rules, for the ReactiveMP engine. The rules approximate the pushforward through `f` by a
method the model names in `DeltaApproximation`:

- `Unscented()` and `Linearization()`, from MessagePassingRulesApproximations, for normal
  messages;
- `CVIProjection()`, sampling and projection onto an exponential family, for other families; its
  rules are a package extension, loaded with `using ExponentialFamilyProjection`.

A known inverse of `f`, when given, sends the messages towards the inputs through it.

```julia
using DeltaMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, ExponentialFamily

algorithm = DeltaApproximation(method = Linearization(), inverse = y -> (y - 1) / 2)
result = @call_message_update_rule(
    node = DeltaFn, target = (:in, 1), algorithm = algorithm,
    m = (out = NormalMeanVariance(3.0, 2.0), in = (nothing,)),
)
getresult(result)   # NormalMeanVariance(μ = 1.0, v = 0.5)
```

- Documentation: `make docs-delta` from the repository root builds it into `docs/build`; it will
  be published at <https://reactivebayes.github.io/DeltaMessagePassingRules.jl/dev/>.
- Tests: `make test-delta`.
- Depends on MessagePassingRulesBase, MessagePassingRulesApproximations, BayesBase,
  ExponentialFamily and Distributions; ExponentialFamilyProjection is a weak dependency, for
  `CVIProjection`. Julia 1.11 or later. MIT licence.
