# ProbitMessagePassingRules

The probit node for reactive message passing, `out ~ Bernoulli(Φ(in))` with `Φ` the standard
normal CDF: a binary observation, or the probability of a `1`, through a univariate normal
latent input. Its own algorithm, `ProbitEP`, is expectation propagation, which keeps the message
towards `in` normal; plain rules under `DefaultAlgorithm` give the exact likelihood instead.

```julia
using ProbitMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

result = @call_message_update_rule(node = Probit, target = :out, m = (in = NormalMeanVariance(1.0, 0.5),))
getresult(result)   # Bernoulli(Φ(1 / √1.5)) ≈ Bernoulli(0.793)

result = @call_message_update_rule(node = Probit, target = :in, m = (out = PointMass(1.0), in = NormalMeanPrecision(0.0, 1.0)))
getresult(result)   # a NormalWeightedMeanPrecision, by expectation propagation
```

- Documentation: `make docs-probit` from the repository root builds it into `docs/build`; it
  will be published at <https://reactivebayes.github.io/ProbitMessagePassingRules.jl/dev/>.
- Tests: `make test-probit`.
- Depends on MessagePassingRulesBase, MessagePassingRulesApproximations (Gauss–Hermite
  cubature for the average energy), BayesBase, ExponentialFamily, Distributions and StatsFuns.
  MIT licence.
