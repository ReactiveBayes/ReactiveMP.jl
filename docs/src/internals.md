# [Internals](@id internals)

The helpers on this page are internal: the engine's own machinery, documented for contributors.
They are not part of the public API, and they may change without notice.

```@setup internals
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

## [Node inputs and dependencies](@id internals-dependencies)

At activation, a node finds what each outbound message is computed from. These inputs are the
[dependencies](@extref MessagePassingRulesBase glossary-dependencies) its algorithm declares, or
else the engine's [default scheme](@extref MessagePassingRulesBase glossary-default-scheme).
[`ReactiveMP.default_dependencies`](@ref) computes the default scheme for one interface. Here it
runs for the interface `μ` (position 2) of the `Gaussian` node of
[the example node](@ref example-node), under three factorisations:

```@example internals
x, y = randomvar(label = :x), datavar(label = :y)
interfaces = [(:out, y), (:μ, x), (:v, constvar(1.0))]

map([((:out, :μ, :v),), ((:out,), (:μ,), (:v,)), ((:out, :μ), (:v,))]) do factorisation
    messages, marginals = ReactiveMP.default_dependencies(factornode(Gaussian, interfaces, factorisation), 2)
    (messages = first(messages), marginals = first(marginals))
end
```

With one cluster, the rule towards `μ` reads the messages on `out` and `v`. With one cluster per
interface, it reads their marginals. With the clusters `(out, μ)` and `(v)`, it reads the message
on `out` and the marginal of `v`.

A rule runs once every input has a new value since its last run, so each of its inputs refreshes
once per run, which in variational message passing is one coordinate update. Its marginal inputs
make one exception, [`ReactiveMP.RelaxOnce`](@ref): after a run that consumed initial marginals
alone, a single refreshed input runs it once more. That starts structured factorisations whose
clusters depend on each other, which would otherwise wait on each other forever, without letting
rules re-fire each other on initial values before any data arrives.

Each input carries a label for the rule: its interface name, a cluster's key, or a group member.
[`ReactiveMP.input_names`](@ref) folds a group's members into one tuple, and the labels become the
names a [`ReactiveMP.MessageMapping`](@ref) carries.

```@docs
ReactiveMP.getalgorithm
ReactiveMP.default_dependencies
ReactiveMP.declared_dependencies
ReactiveMP.extended_default_dependencies
ReactiveMP.rule_target
ReactiveMP.input_label
ReactiveMP.input_names
ReactiveMP.GroupMember
ReactiveMP.GroupInputs
ReactiveMP.EmptyGroup
ReactiveMP.FactorNodeLocalClusters
ReactiveMP.clusterkey
ReactiveMP.RelaxOnce
```

## [Rule arguments](@id internals-rule-arguments)

A mapping hands a rule its inputs through [`ReactiveMP.rule_arguments`](@ref). It builds them from
the latest messages and marginals with the helpers below, and it collects their annotations the
same way.

```@docs
ReactiveMP.rule_messages
ReactiveMP.rule_marginals
ReactiveMP.rule_annotations
```

## [Scratch](@id internals-scratch)

A rule may declare a [scratch](@extref MessagePassingRulesBase glossary-scratch), working memory
that it writes before it reads. Each message and marginal mapping keeps the scratch of the rule it
runs. The mapping builds the scratch at the rule's first call and reuses it while the same rule
runs on the stream. The scratch never leaves the rule and is never shared with another.

```@docs
ReactiveMP.ScratchSlot
ReactiveMP.scratch_for!
ReactiveMP.poison!
```

## [Static inputs](@id internals-static-inputs)

A node with static inputs reads some of its interfaces as values, not as messages: a constant, or
the last observation of a data variable. [`ReactiveMP.StaticFold`](@ref) passes those values to
the node's function, and [`ReactiveMP.with_statics`](@ref) holds a node's streams back until
every static input has a value (see [Static inputs](@ref lib-node-static-inputs)).

```@docs
ReactiveMP.StaticFold
ReactiveMP.with_statics
```

## [The equality chain](@id internals-equality)

A random variable with more than one connection computes its outbound messages along an equality
chain. The chain has one equality node per connection, and the nodes cache the partial products of
the variable's inbound messages.

```@docs
ReactiveMP.EqualityChain
ReactiveMP.EqualityNode
```

## [Callbacks](@id internals-callbacks)

[`ReactiveMP.merge_callbacks`](@ref) returns a [`ReactiveMP.MergedCallbacks`](@ref), and the
engine reports its events through [`ReactiveMP.@invoke_callback`](@ref), which builds an event
only for a handler that listens to it (see [Callbacks](@ref lib-callbacks)).

```@docs
ReactiveMP.MergedCallbacks
ReactiveMP.@invoke_callback
```

## [Helpers](@id internals-helpers)

`@proxy_methods` defines the statistics, such as `mean` and `var`, that [`Message`](@ref) and
[`Marginal`](@ref) forward to their data.

```@docs
ReactiveMP.MacroHelpers.@proxy_methods
```
