export getannotations

"""
    AnnotationDict()
    AnnotationDict(other::AnnotationDict)

A mutable dictionary that associates `Symbol` keys with arbitrary annotation values.
Supports lazy initialization — no memory is allocated until the first write.

The copy constructor creates an independent shallow copy of `other`.
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

# Overloaded later for `::Message` and `::Marginal` in their respective files
function getannotations end

Base.isempty(ann::AnnotationDict) =
    isnothing(ann.data) || isempty(ann.data::Dict{Symbol, Any})

function Base.show(io::IO, ann::AnnotationDict)
    if isempty(ann)
        print(io, "AnnotationDict()")
    else
        print(io, "AnnotationDict(")
        join(io, ("$k => $v" for (k, v) in ann.data::Dict{Symbol, Any}), ", ")
        print(io, ")")
    end
end

function Base.:(==)(left::AnnotationDict, right::AnnotationDict)
    return left.data == right.data
end

"""
    has_annotation(ann::AnnotationDict, key::Symbol) -> Bool

Return `true` if `ann` contains an entry for `key`, `false` otherwise.
"""
function has_annotation(ann::AnnotationDict, key::Symbol)
    return !isnothing(ann.data) && haskey(ann.data::Dict{Symbol, Any}, key)
end

"""
    annotate!(ann::AnnotationDict, key::Symbol, value)

Store `value` under `key` in `ann`. Always returns `nothing`.
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
    get_annotation(ann::AnnotationDict, key::Symbol)

Return the value stored under `key`. Throws `KeyError` if `key` is absent.
"""
function get_annotation(ann::AnnotationDict, key::Symbol)
    if isnothing(ann.data)
        throw(KeyError(key))
    end
    return (ann.data::Dict{Symbol, Any})[key]
end

"""
    get_annotation(ann::AnnotationDict, ::Type{T}, key::Symbol) where {T}

Return the value stored under `key`, converted to type `T`. Throws `KeyError` if
`key` is absent.
"""
function get_annotation(ann::AnnotationDict, ::Type{T}, key::Symbol) where {T}
    return convert(T, get_annotation(ann, key))::T
end

"""
    AbstractAnnotations

Abstract base type for annotation processors. Subtypes define how annotations
are written into messages after rule execution and merged during message products.

See also: [`post_product_annotations!`](@ref), [`post_rule_annotations!`](@ref)
"""
abstract type AbstractAnnotations end

"""
    post_product_annotations!(processor::AbstractAnnotations, merged::AnnotationDict, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, left_dist, right_dist)

Write annotations into `merged` based on `left_ann`, `right_ann`, and the distributions
involved in the message product. Called once per processor inside
[`compute_product_of_two_messages`](@ref).
"""
function post_product_annotations! end

"""
    pre_rule_annotations!(processor::AbstractAnnotations, ann::AnnotationDict, mapping, messages, marginals)

Write annotations into `ann` before a rule has executed. Called once per processor
inside the `MessageMapping` callable, before the rule returns its result distribution.
"""
function pre_rule_annotations! end

"""
    post_rule_annotations!(processor::AbstractAnnotations, ann::AnnotationDict, mapping, messages, marginals, result)

Write annotations into `ann` after a rule has executed. Called once per processor
inside the `MessageMapping` callable, after the rule returns its result distribution.
"""
function post_rule_annotations! end

"""
    post_product_annotations!(processors, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, left_dist, right_dist) -> AnnotationDict

Produce a merged `AnnotationDict` from the annotations of two messages being multiplied.
Called inside [`compute_product_of_two_messages`](@ref).

If `left_dist` is `missing` the right annotations are copied through unchanged, and vice versa.
If both are `missing`, or if `processors` is `nothing`, an empty `AnnotationDict` is returned.
Otherwise each processor in `processors` is called via the per-processor `post_product_annotations!`
to populate the result.
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

post_product_annotations!(processors, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, ::Missing, ::Missing) = AnnotationDict()

post_product_annotations!(processors, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, ::Missing, right_dist) = AnnotationDict(
    right_ann
)

post_product_annotations!(processors, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, left_dist, ::Missing) = AnnotationDict(
    left_ann
)
