# [Marginals](@id lib-marginal)

A marginal is a belief: about one variable, the normalised product of the messages arriving at
it, or about a cluster of a factor node's variables, the node's local joint marginal. Marginals
are what inference is run for, what variational rules read, and what the free energy is computed
from.

## [The marginal type](@id lib-marginal-type)

Every marginal is a [`Marginal`](@ref). Like a [`Message`](@ref), it holds its data and forwards
the statistics to it, and records whether it is clamped and whether it is initial.

```@example marginal
using ReactiveMP, BayesBase, ExponentialFamily

marginal = Marginal(NormalMeanPrecision(0.0, 1.0), false, true)
mean(marginal), precision(marginal), is_initial(marginal)
```

```@docs
Marginal
getdata(::Marginal)
is_clamped(::Marginal)
is_initial(::Marginal)
getannotations(::Marginal)
getlogscale(::Marginal)
as_marginal
```

## [A variable's marginal](@id lib-marginal-variable)

A random variable's marginal is the product of its inbound messages, formed with
[`as_marginal`](@ref) in its public type: a rule may compute in an efficient working type, such
as `WishartFast`, which
[`public_equivalent`](@extref MessagePassingRulesBase.public_equivalent) turns into the type users
expect, `Wishart`, before the marginal leaves the product. It keeps the product's annotations and,
when log scales are tracked, its log scale: in a tree-shaped model inferred exactly by belief
propagation, the log evidence of the data. A data variable's marginal is its latest observation,
and a constant's the constant, both with log scale zero.

## [Joint marginals](@id lib-marginal-joint)

A cluster of more than one interface of a factor node has a joint marginal, computed by the node's
marginal rule from the messages on the cluster's interfaces and the marginals of the other
clusters, by a [`ReactiveMP.MarginalMapping`](@ref). A structured variational rule reads it as
`q[:out, :μ]`, and the free energy uses it. A joint whose parts are independent, such as a
cluster with an observed member, may be a
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) of blocks, which the rules
read block by block. A joint marginal carries no annotations and no log scale.

```@docs
ReactiveMP.MarginalMapping
```

## [Marginal streams](@id lib-marginal-observable)

Like messages, marginals live as streams, [`ReactiveMP.MarginalObservable`](@ref)s, which emit a
new belief whenever the messages or marginals it is computed from change. Every variable holds
one, read with [`ReactiveMP.get_stream_of_marginals`](@ref); a factor node holds one per joint
cluster. The stream is lazy until activation connects it; before that,
[`ReactiveMP.set_initial_marginal!`](@ref) can seed it, so that a rule that depends on the marginal
at the start has something to read. The latest marginal is kept: a subscriber that joins late
receives it at once.

```@docs
ReactiveMP.MarginalObservable
```
