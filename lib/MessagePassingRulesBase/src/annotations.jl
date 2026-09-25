"""
    NoAnnotations()

Annotations that are never recorded. Zero fields, so a rule that does not annotate pays
nothing.
"""
struct NoAnnotations end

"""
    AnnotationStore()

A mutable set of annotations, keyed by symbol. The storage is created on first write.
"""
mutable struct AnnotationStore
    entries::Union{Nothing, Dict{Symbol, Any}}
end

AnnotationStore() = AnnotationStore(nothing)

"""
    annotate!(annotations, key::Symbol, value)

Record `value` under `key`.
"""
function annotate!(store::AnnotationStore, key::Symbol, value)
    entries = store.entries === nothing ? (store.entries = Dict{Symbol, Any}()) : store.entries
    entries[key] = value
    return nothing
end
annotate!(::NoAnnotations, ::Symbol, _) = nothing

"""
    hasannotation(annotations, key::Symbol)

Whether `annotations` holds a value under `key`; always `false` for `NoAnnotations`.
"""
hasannotation(store::AnnotationStore, key::Symbol) = store.entries !== nothing && haskey(store.entries, key)
hasannotation(::NoAnnotations, ::Symbol) = false

"""
    getannotation(annotations, key::Symbol[, default])

The annotation recorded under `key`, or `default`. Without a default, a missing key throws
a `KeyError`.
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

The `ann` a rule body receives, carrying annotations in both directions. The ones that
arrived with the inputs are keyed exactly like them, `ann.m[:out]` and `ann.q[:μ]`; the rule
writes its own with `annotate!(ann, key, value)`, which lands in `out`. Annotations never
take part in dispatch.
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
