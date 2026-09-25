
# [Helper utilities](@id lib-helpers)

This page documents utility types and functions of the engine that code around it may use.

## [Iteration helpers](@id lib-helpers-iteration)

When a message update rule computes the outgoing message on edge `k` of a factor node, it needs the incoming messages from *all other edges* — every edge except `k`. The [`ReactiveMP.SkipIndexIterator`](@ref) provides an allocation-free view of a collection that skips one index.

The constructor [`skipindex`](@ref) is the standard way to create one:

```julia
# messages is a length-3 collection; compute outgoing message for edge 2
# by iterating over edges 1 and 3 only
other = ReactiveMP.skipindex(messages, 2)
collect(other)   # [messages[1], messages[3]]
```

```@docs
ReactiveMP.SkipIndexIterator
ReactiveMP.skipindex
```

## [Macro utilities](@id lib-helpers-macro)

`ReactiveMP.MacroHelpers.@proxy_methods` generates forwarding methods, so that a thin wrapper type
delegates calls to what it wraps: `Message` and `Marginal` forward `mean`, `var` and the other
statistics to their data.

```@docs
ReactiveMP.MacroHelpers.@proxy_methods
```
