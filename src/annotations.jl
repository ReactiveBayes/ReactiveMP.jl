export getannotations

import MessagePassingRulesBase: annotate!

"""
    ReactiveMP.AnnotationDict()
    ReactiveMP.AnnotationDict(other::AnnotationDict)

The annotations of a message or a marginal: values keyed by `Symbol`, metadata about how it was
computed. It allocates nothing until the first write. The second form is a shallow copy of
`other`.

A rule reads and writes it through the base package's functions,
[`getannotation`](@extref MessagePassingRulesBase.getannotation),
[`hasannotation`](@extref MessagePassingRulesBase.hasannotation) and
[`annotate!`](@extref MessagePassingRulesBase.annotate!); the engine's own are
[`ReactiveMP.get_annotation`](@ref), [`ReactiveMP.has_annotation`](@ref) and
[`ReactiveMP.annotate!`](@ref ReactiveMP.annotate!(::ReactiveMP.AnnotationDict, ::Symbol, ::Any)).

# Examples

```jldoctest
julia> ann = ReactiveMP.AnnotationDict();

julia> ReactiveMP.annotate!(ann, :count, 1);

julia> ReactiveMP.has_annotation(ann, :count), ReactiveMP.get_annotation(ann, :count)
(true, 1)
```
"""
mutable struct AnnotationDict
    data::Union{Nothing, Dict{Symbol, Any}}

    function AnnotationDict()
        return new(nothing)
    end

    function AnnotationDict(other::AnnotationDict)
        return new(
            if isnothing(other.data)
                nothing
            else
                copy(other.data::Dict{Symbol, Any})
            end,
        )
    end
end

# `MessagePassingRulesBase`'s, overloaded later for `::Message` and `::Marginal` in their respective files
import MessagePassingRulesBase: getannotations

Base.isempty(ann::AnnotationDict) =
    isnothing(ann.data) || isempty(ann.data::Dict{Symbol, Any})

function Base.show(io::IO, ann::AnnotationDict)
    if isempty(ann)
        print(io, "AnnotationDict()")
        return nothing
    end
    data = ann.data::Dict{Symbol, Any}
    if get(io, :compact, false)
        print(io, "AnnotationDict(n=", length(data), ")")
    else
        print(io, "AnnotationDict(")
        join(io, ("$k => $v" for (k, v) in data), ", ")
        print(io, ")")
    end
    return nothing
end

function Base.:(==)(left::AnnotationDict, right::AnnotationDict)
    return left.data == right.data
end

"""
    ReactiveMP.has_annotation(ann::AnnotationDict, key::Symbol) -> Bool

Whether `ann` holds a value under `key`.
"""
function has_annotation(ann::AnnotationDict, key::Symbol)
    return !isnothing(ann.data) && haskey(ann.data::Dict{Symbol, Any}, key)
end

"""
    ReactiveMP.annotate!(ann::AnnotationDict, key::Symbol, value) -> Nothing

Store `value` under `key` in `ann`, replacing any value already there.
"""
function annotate!(ann::AnnotationDict, key::Symbol, value)
    if isnothing(ann.data)
        data = Dict{Symbol, Any}(key => value)
        ann.data = data
    else
        (ann.data::Dict{Symbol, Any})[key] = value
    end
    return nothing
end

"""
    ReactiveMP.get_annotation(ann::AnnotationDict, key::Symbol)
    ReactiveMP.get_annotation(ann::AnnotationDict, ::Type{T}, key::Symbol) -> T

The value stored under `key` in `ann`; with a type `T`, converted to `T`.

# Throws

- `KeyError` when `ann` holds nothing under `key`.
"""
function get_annotation(ann::AnnotationDict, key::Symbol)
    if isnothing(ann.data)
        throw(KeyError(key))
    end
    return (ann.data::Dict{Symbol, Any})[key]
end

function get_annotation(ann::AnnotationDict, ::Type{T}, key::Symbol) where {T}
    return convert(T, get_annotation(ann, key))::T
end

# A rule reads and writes annotations through the base package's functions.
MessagePassingRulesBase.hasannotation(ann::AnnotationDict, key::Symbol) = has_annotation(ann, key)
MessagePassingRulesBase.getannotation(ann::AnnotationDict, key::Symbol) = get_annotation(ann, key)
MessagePassingRulesBase.getannotation(ann::AnnotationDict, key::Symbol, default) =
    has_annotation(ann, key) ? get_annotation(ann, key) : default

"""
    ReactiveMP.AbstractAnnotations

The supertype of annotation processors, which write annotations on messages. A processor
implements three hooks:

- [`ReactiveMP.pre_rule_annotations!`](@ref), before a message rule runs;
- [`ReactiveMP.post_rule_annotations!`](@ref), after it ran;
- [`ReactiveMP.post_product_annotations!`](@ref), when two messages are multiplied, to merge their
  annotations into the product's.

Processors reach the rules through the activation option `annotations` of
[`ReactiveMP.FactorNodeActivationOptions`](@ref), and the products through the `annotations` of a
variable's [`ReactiveMP.MessageProductContext`](@ref): a processor is given to both. The built-in
processor is [`InputArgumentsAnnotations`](@ref).
"""
abstract type AbstractAnnotations end

"""
    ReactiveMP.post_product_annotations!(processor::AbstractAnnotations, merged::AnnotationDict, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, left_dist, right_dist)

The hook of `processor` at a product of two messages: write into `merged`, the product's empty
annotations, from the two messages' annotations and their distributions. Its return value is
ignored.
"""
function post_product_annotations! end

"""
    ReactiveMP.pre_rule_annotations!(processor::AbstractAnnotations, ann::AnnotationDict, mapping, messages, marginals)

The hook of `processor` before a message rule runs: write into `ann`, the new message's
annotations, from the [`ReactiveMP.MessageMapping`](@ref) and the rule's inbound messages and
marginals. It runs for a `missing` message too. Its return value is ignored.
"""
function pre_rule_annotations! end

"""
    ReactiveMP.post_rule_annotations!(processor::AbstractAnnotations, ann::AnnotationDict, mapping, messages, marginals, result)

The hook of `processor` after a message rule ran: write into `ann`, the new message's
annotations, from the [`ReactiveMP.MessageMapping`](@ref), the rule's inputs and `result`, the
message's data. It is skipped for a `missing` message, when no rule ran. Its return value is
ignored.
"""
function post_rule_annotations! end

"""
    ReactiveMP.post_product_annotations!(processors, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, left_dist, right_dist) -> AnnotationDict

The annotations of the product of two messages, which
[`ReactiveMP.compute_product_of_two_messages`](@ref) computes: a new
[`ReactiveMP.AnnotationDict`](@ref) that each processor writes into with its own
`post_product_annotations!`. When one side's distribution is `missing`, the product is the other
side, and its annotations are that side's, copied, whatever the processors; when both are
`missing`, or `processors` is `nothing`, they are empty.
"""
function post_product_annotations!(
        processors,
        left_ann::AnnotationDict,
        right_ann::AnnotationDict,
        new_dist,
        left_dist,
        right_dist,
    )
    merged = AnnotationDict()
    if isnothing(processors)
        return merged
    end
    for p in processors
        post_product_annotations!(
            p, merged, left_ann, right_ann, new_dist, left_dist, right_dist
        )
    end
    return merged
end

post_product_annotations!(
    processors,
    left_ann::AnnotationDict,
    right_ann::AnnotationDict,
    new_dist,
    ::Missing,
    ::Missing,
) = AnnotationDict()

post_product_annotations!(
    processors,
    left_ann::AnnotationDict,
    right_ann::AnnotationDict,
    new_dist,
    ::Missing,
    right_dist,
) = AnnotationDict(right_ann)

post_product_annotations!(
    processors,
    left_ann::AnnotationDict,
    right_ann::AnnotationDict,
    new_dist,
    left_dist,
    ::Missing,
) = AnnotationDict(left_ann)
