# [Marginals](@id lib-marginal)

A [marginal](@extref MessagePassingRulesBase glossary-marginal) is a belief. The marginal of a
variable is the normalised product of the messages that arrive at it. The marginal of a
[cluster](@extref MessagePassingRulesBase glossary-cluster) of a factor node is the node's local
joint marginal. You run inference for the marginals, variational rules read them, and the
[free energy](@ref lib-score) is computed from them.

```@setup marginal
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node).

## [The marginal type](@id lib-marginal-type)

Every marginal is a [`Marginal`](@ref). Like a [`Message`](@ref), it holds its data and forwards
the statistics to it, and it records whether it is clamped and whether it is initial:

```@example marginal
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
ReactiveMP.skip_initial
```

## [A variable's marginal](@id lib-marginal-variable)

A random variable's marginal is the product of its inbound messages, formed with
[`as_marginal`](@ref) in its public type. A rule may compute in an efficient working type, such as
`WishartFast`, which [`public_equivalent`](@extref MessagePassingRulesBase.public_equivalent)
turns into the type users expect, `Wishart`, before the marginal leaves the product. The marginal
keeps the product's annotations. When log scales are tracked, it also keeps the product's log
scale: in a tree-shaped model inferred exactly by belief propagation, the log evidence of the
data.

A data variable's marginal is its latest observation, and a constant's is the constant, both
with log scale zero.

## [Joint marginals](@id lib-marginal-joint)

A cluster of more than one interface of a factor node has a joint marginal. A
[`ReactiveMP.MarginalMapping`](@ref) computes it with the node's marginal rule, from the messages
on the cluster's interfaces and the marginals of the other clusters. A structured variational
rule reads it as `q[:out, :μ]`, and the free energy uses it.

```@example marginal
x, y = randomvar(label = :x), datavar(label = :y)
prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
foreach(n -> activate!(n, FactorNodeActivationOptions()), (prior, likelihood))

joint = only(ReactiveMP.get_node_local_marginals(ReactiveMP.getlocalclusters(likelihood)))
joints = Marginal[]
subscription = subscribe!(get_stream_of_marginals(joint), (q) -> push!(joints, q))
new_observation!(y, 2.0)
last(joints)
```

The likelihood's cluster is `(out, μ, v)`. Its `out` is observed and its `v` constant, so the
joint factorises: a [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) of
blocks, which rules read block by block. A joint marginal carries no annotations and no log
scale.

```@docs
ReactiveMP.MarginalMapping
```

## [Marginal streams](@id lib-marginal-observable)

Like messages, marginals live as streams, [`ReactiveMP.MarginalObservable`](@ref)s. A stream emits
a new belief whenever the messages or marginals it is computed from change. Every variable holds
one, which [`ReactiveMP.get_stream_of_marginals`](@ref) returns, and a factor node holds one per
joint cluster. The stream is lazy until activation connects it. Before that,
[`ReactiveMP.set_initial_marginal!`](@ref) can seed it, so that a rule that depends on the
marginal at the start has something to read. The stream keeps its latest marginal: a subscriber
that joins late receives it at once.

```@docs
ReactiveMP.MarginalObservable
```
