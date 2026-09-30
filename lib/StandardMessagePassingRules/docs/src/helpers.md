# Helper nodes and message types

This page covers two nodes that are not a distribution family with parameters, a message type
for the shape of a Gamma, and a constructor that models use. The two nodes send a fixed
[message](@extref MessagePassingRulesBase glossary-message) towards `out`, which is exact under
any [factorisation](@extref MessagePassingRulesBase glossary-factorisation).

```@setup helpers
using MessagePassingRulesBase, StandardMessagePassingRules, Distributions, BayesBase
```

## StandaloneDistribution

[`StandaloneDistribution`](@ref) makes a fixed distribution a factor. Its interface
`distribution` holds the distribution as a constant, and the message towards `out` is that
distribution.

```math
f(\mathrm{out}, d) = d(\mathrm{out})
```

```@example helpers
MessagePassingRulesBase.rule_coverage(StandaloneDistribution)
```

The rule reads the constant as a [point-mass](@extref MessagePassingRulesBase glossary-point-mass)
[marginal](@extref MessagePassingRulesBase glossary-marginal) that holds the distribution. The
card calls the rule variational because it takes a marginal, but the message is the distribution
itself:

```@example helpers
@call_message_update_rule(node = StandaloneDistribution, target = :out, q = (distribution = PointMass(Beta(2.0, 3.0)),))
```

**Limitations.** No rule towards `distribution`, which must be a constant.

```@docs
StandaloneDistribution
```

## Uninformative

[`Uninformative`](@ref) is the factor of one. Its message is `Uninformative()`, which leaves
every product unchanged.

```math
f(\mathrm{out}) = 1
```

```@example helpers
MessagePassingRulesBase.rule_coverage(Uninformative)
```

**Limitations.** `Uninformative()` is not a distribution, and its message declares no
[log scale](@extref MessagePassingRulesBase glossary-log-scale).

```@docs
Uninformative
```

## GammaShapeLikelihood

[`GammaShapeLikelihood`](@ref) is the message towards the shape of a Gamma, from
`GammaShapeRate` and from [`GammaMixture`](@ref). ExponentialFamily has no closed-form conjugate
prior for a Gamma's shape, so the message is a type of its own.

```@docs
GammaShapeLikelihood
```

## diageye

`diageye(n)` returns the `n`×`n` identity as a dense matrix. It is MessagePassingRulesBase's
[`diageye`](@extref MessagePassingRulesBase.diageye), and this package exports it so that models
can write covariances and precisions with it.
