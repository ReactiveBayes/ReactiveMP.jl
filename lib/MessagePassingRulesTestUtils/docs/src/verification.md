# Verification against the node

A table's expected values are derived by hand. Verification checks a message rule against the
thing it is derived from, the node's density: the message a rule computes must be, up to a
constant, the node's factor integrated against its other inputs, which the verification computes
numerically at a set of points.

With messages as inputs, the reference is belief propagation,

```math
\log \mu(x) = \log \int f(x, y_1, \dots, y_n) \prod_j m_j(y_j) \, dy + \text{const},
```

and the constant is the rule's log scale: the message times its exponential is the integral
itself. With marginals, it is naive variational message passing,
``\log \mu(x) = \mathrm{E}_{q}[\log f(x, y)] + \text{const}``.

```julia
@verify_message_update_rule(node = NormalMeanVariance, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
@verify_message_update_rule(node = NormalMeanVariance, target = :μ, q = (out = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)))
```

## Limitations

- The node must be stochastic and without groups: the reference is its
  [`nodefunction`](@extref MessagePassingRulesBase.nodefunction).
- The inputs are point masses, which are substituted, finite discrete distributions, which are
  enumerated, and at most two continuous univariate distributions, which are integrated.
- The message is univariate, compared at its quantiles or over its support.
- A case takes messages or marginals, not both, so a rule mixing them is checked by its table
  only.

[`verify_message_update`](@ref) runs the same check on any function returning a message and its
log scale, for an implementation that is not a rule of the base package.

## API

```@docs
@verify_message_update_rule
verify_message_update_rule
verify_message_update
```
