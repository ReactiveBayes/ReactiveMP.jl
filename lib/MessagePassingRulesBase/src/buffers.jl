"""
    buffer_like(x)
    buffer_like(x::AbstractArray, T::Type)

Uninitialised storage shaped like `x`, with element type `T` if given, for an in-place rule's
`preallocate`. It dispatches on the type of `x`, so the buffer is of a kind that matches:
`similar` for arrays, which keeps a static array static-sized and mutable and a device array on
its device; elementwise for tuples and named tuples. Array types and devices that need something
else add a method.

# Throws
`ArgumentError` for a number, which has no storage to write into.

# Examples

```jldoctest
julia> using MessagePassingRulesBase: buffer_like

julia> b = buffer_like([1.0, 2.0]); (typeof(b), size(b))
(Vector{Float64}, (2,))
```
"""
buffer_like(x::AbstractArray) = similar(x)
buffer_like(x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
buffer_like(x::Tuple) = map(buffer_like, x)
buffer_like(x::NamedTuple) = map(buffer_like, x)
buffer_like(x::Number) = throw(ArgumentError("a number has no storage to write into; an in-place rule needs a mutable output"))
