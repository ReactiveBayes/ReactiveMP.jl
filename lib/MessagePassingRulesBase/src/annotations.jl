"""
    NoAnnotations()

An annotation store that records nothing: [`annotate!`](@ref) on it does nothing,
[`hasannotation`](@ref) is always `false`, and [`getannotation`](@ref) finds nothing. It has no
fields, so a call that collects no annotations pays nothing for the rule's writes. It is the
default for the `message_passing_*` calls, and the `out` of a [`RuleAnnotations`](@ref) unless
one is given.
"""
struct NoAnnotations end

"""
    AnnotationStore()

A mutable store of annotations keyed by symbol, which collects what a rule writes with
[`annotate!`](@ref). Its dictionary is created on the first write, so an empty store allocates
nothing more. Pass one as a call's `ann` to read afterwards what the rule annotated, through
[`getannotations`](@ref)`(result)`.

```jldoctest
julia> using MessagePassingRulesBase: AnnotationStore, annotate!, hasannotation, getannotation

julia> store = AnnotationStore(); annotate!(store, :iterations, 3);

julia> hasannotation(store, :iterations), getannotation(store, :iterations), getannotation(store, :other, 0)
(true, 3, 0)
```

See also [`NoAnnotations`](@ref), [`RuleAnnotations`](@ref).
"""
mutable struct AnnotationStore
    entries::Union{Nothing, Dict{Symbol, Any}}
end

AnnotationStore() = AnnotationStore(nothing)

"""
    annotate!(annotations, key::Symbol, value) -> nothing

Record `value` under `key`, replacing any earlier value. `annotations` is an
[`AnnotationStore`](@ref), a [`NoAnnotations`](@ref), which drops it, or a rule's `ann`, a
[`RuleAnnotations`](@ref), which writes to its `out`. A rule body calls it as
`annotate!(ann, key, value)`.
"""
function annotate!(store::AnnotationStore, key::Symbol, value)
    entries = store.entries === nothing ? (store.entries = Dict{Symbol, Any}()) : store.entries
    entries[key] = value
    return nothing
end
annotate!(::NoAnnotations, ::Symbol, _) = nothing

"""
    hasannotation(annotations, key::Symbol) -> Bool

Whether `annotations` holds a value under `key`; always `false` for a [`NoAnnotations`](@ref).
On a rule's `ann` it looks at what the rule wrote, not at the inputs' annotations.
"""
hasannotation(store::AnnotationStore, key::Symbol) = store.entries !== nothing && haskey(store.entries, key)
hasannotation(::NoAnnotations, ::Symbol) = false

"""
    getannotation(annotations, key::Symbol)
    getannotation(annotations, key::Symbol, default)

The annotation recorded under `key`, or `default` when there is none. On a rule's `ann` it
reads what the rule wrote, not the inputs' annotations.

# Throws
`KeyError` when `key` holds nothing and no `default` is given.
"""
function getannotation(store::AnnotationStore, key::Symbol)
    hasannotation(store, key) || throw(KeyError(key))
    return store.entries[key]
end
getannotation(store::AnnotationStore, key::Symbol, default) =
    hasannotation(store, key) ? store.entries[key] : default
getannotation(::NoAnnotations, key::Symbol) = throw(KeyError(key))
getannotation(::NoAnnotations, ::Symbol, default) = default

"""
    RuleAnnotations(; m = NamedTuple(), q = NamedTuple(), out = NoAnnotations())

The `ann` a rule body receives, carrying annotations in both directions. The ones that arrived
with the inputs are keyed exactly like them, `ann.m[:out]` and `ann.q[:μ]`; the rule writes its
own with [`annotate!`](@ref)`(ann, key, value)`, which lands in `out`, and reads them back with
[`hasannotation`](@ref) and [`getannotation`](@ref). Annotations never take part in dispatch.

# Keywords
- `m`: the annotations of the inbound messages, a `NamedTuple` keyed like `args.m`. Default: none.
- `q`: the annotations of the marginals, keyed like `args.q`. Default: none.
- `out`: where the rule's own annotations go, an [`AnnotationStore`](@ref) or a
  [`NoAnnotations`](@ref). Default: `NoAnnotations()`, dropping them.
"""
struct RuleAnnotations{M <: Messages, Q <: Marginals, O}
    m::M
    q::Q
    out::O
end

RuleAnnotations(; m = NamedTuple(), q = NamedTuple(), out = NoAnnotations()) =
    RuleAnnotations(as_messages(m), as_marginals(q), out)

annotate!(ann::RuleAnnotations, key::Symbol, value) = annotate!(ann.out, key, value)
hasannotation(ann::RuleAnnotations, key::Symbol) = hasannotation(ann.out, key)
getannotation(ann::RuleAnnotations, key::Symbol) = getannotation(ann.out, key)
getannotation(ann::RuleAnnotations, key::Symbol, default) = getannotation(ann.out, key, default)
