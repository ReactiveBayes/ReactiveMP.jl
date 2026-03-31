
"""
    AnnotationDict

A mutable dictionary that associates `Symbol` keys with arbitrary annotation values.
Supports lazy initialization — no memory is allocated until the first write.
"""
mutable struct AnnotationDict
    data::Union{Nothing, Dict{Symbol, Any}}

    function AnnotationDict()
        return new(nothing)
    end
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
