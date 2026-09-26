# Helper nodes and message types

Two nodes that are not a distribution family with parameters, a message type of the Gamma
shape, and a constructor models use.

```@setup helpers
using MessagePassingRulesBase, StandardMessagePassingRules
```

## StandaloneDistribution

A fixed distribution as a factor: its interface `distribution` holds the distribution as a
constant, and the message towards `out` is that distribution.

```math
f(\mathrm{out}, d) = d(\mathrm{out})
```

```@example helpers
MessagePassingRulesBase.rule_coverage(StandaloneDistribution)
```

**Limitations.** No rule towards `distribution`, which must be a constant.

```@docs
StandaloneDistribution
```

## Uninformative

The factor of one: its message is `Uninformative()`, which every product leaves unchanged.

```math
f(\mathrm{out}) = 1
```

```@example helpers
MessagePassingRulesBase.rule_coverage(Uninformative)
```

**Limitations.** `Uninformative()` is not a distribution, and its message declares no log scale.

```@docs
Uninformative
```

## GammaShapeLikelihood

The message towards the shape of a Gamma, from `GammaShapeRate` and from [`GammaMixture`](@ref).
It has no closed-form conjugate prior in ExponentialFamily, so it is a type of its own.

```@docs
GammaShapeLikelihood
```

## diageye

```@docs
diageye
```
