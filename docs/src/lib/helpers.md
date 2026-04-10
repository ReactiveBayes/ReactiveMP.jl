
# [Helper utilities](@id lib-helpers)

This page documents utility types and functions that appear at the boundaries of the inference engine — primarily in custom node and rule implementations. They are not needed for everyday inference with built-in nodes, but become useful when writing [`@rule`](@ref) definitions or building new factor node types.

## [Iteration helpers](@id lib-helpers-iteration)

When a message update rule computes the outgoing message on edge `k` of a factor node, it needs the incoming messages from *all other edges* — every edge except `k`. The [`ReactiveMP.SkipIndexIterator`](@ref) provides an allocation-free view of a collection that skips one index.

The constructor [`skipindex`](@ref) is the standard way to create one:

```julia
# messages is a length-3 collection; compute outgoing message for edge 2
# by iterating over edges 1 and 3 only
other = ReactiveMP.skipindex(messages, 2)
collect(other)   # [messages[1], messages[3]]
```

This is used internally inside `@rule` dispatch to pass only the relevant inbound messages to the rule computation.

```@docs
ReactiveMP.SkipIndexIterator
ReactiveMP.skipindex
```

## [Macro utilities](@id lib-helpers-macro)

The `ReactiveMP.MacroHelpers` submodule contains building blocks used by the [`@node`](@ref) and [`@rule`](@ref) macros to parse and transform Julia type expressions. These are implementation details of the macro system, but they are documented here for completeness and for users who want to understand or extend the macro infrastructure.

| Function | Purpose |
|----------|---------|
| `ReactiveMP.MacroHelpers.ensure_symbol` | Assert that an expression is a `Symbol`; error otherwise |
| `ReactiveMP.MacroHelpers.bottom_type` | Extract the base type `T` from expressions like `Type{<:T}`, `typeof(T)`, or `T` |
| `ReactiveMP.MacroHelpers.upper_type` | Wrap a type expression into `Type{<:T}` form for dispatch |
| `ReactiveMP.MacroHelpers.proxy_type` | Wrap a type with a proxy type as `ProxyType{<:T}` |
| `ReactiveMP.MacroHelpers.@proxy_methods` | Generate forwarding method definitions for a proxy wrapper type |

`@proxy_methods` is the most user-facing of these. It generates a set of method forwarding stubs so that a thin wrapper type transparently delegates calls to its wrapped type, without hand-writing each delegation.

```@docs
ReactiveMP.MacroHelpers.proxy_type
ReactiveMP.MacroHelpers.ensure_symbol
ReactiveMP.MacroHelpers.@proxy_methods
ReactiveMP.MacroHelpers.upper_type
ReactiveMP.MacroHelpers.bottom_type
```
