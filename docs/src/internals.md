# [Internals](@id internals)

The helpers on this page are internal: the engine's own machinery, documented for contributors.
They are not part of the public API.

## [Node inputs and dependencies](@id internals-dependencies)

At activation, a node finds what each outbound message is computed from: the dependencies its
algorithm declares, or the engine's default scheme. Each input is labelled for the rule, a
group's members folded into one tuple, and the labels become the names a
[`ReactiveMP.MessageMapping`](@ref) carries.

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
```

## [Rule arguments](@id internals-rule-arguments)

A mapping hands a rule its inputs through [`ReactiveMP.rule_arguments`](@ref), built from these:

```@docs
ReactiveMP.rule_messages
ReactiveMP.rule_marginals
ReactiveMP.rule_annotations
```

## [Scratch](@id internals-scratch)

A rule may declare a scratch, working memory it writes before it reads. Each message and marginal
mapping keeps the scratch of the rule it runs, built at the first call and reused while the same
rule runs on the stream; it never leaves the rule and is never shared with another.

```@docs
ReactiveMP.ScratchSlot
ReactiveMP.scratch_for!
ReactiveMP.poison!
```

## [Static inputs](@id internals-static-inputs)

```@docs
ReactiveMP.StaticFold
ReactiveMP.with_statics
```

## [The equality chain](@id internals-equality)

A random variable with more than one connection computes its outbound messages along an equality
chain of equality nodes, one per connection, which cache the partial products of its inbound
messages.

```@docs
ReactiveMP.EqualityChain
ReactiveMP.EqualityNode
```

## [Callbacks](@id internals-callbacks)

```@docs
ReactiveMP.MergedCallbacks
```

## [Helpers](@id internals-helpers)

`skipindex` and `SkipIndexIterator` are exported, and the engine does not use them.
`@proxy_methods` defines the statistics `Message` and `Marginal` forward to their data.

```@docs
ReactiveMP.SkipIndexIterator
skipindex
ReactiveMP.MacroHelpers.@proxy_methods
```
