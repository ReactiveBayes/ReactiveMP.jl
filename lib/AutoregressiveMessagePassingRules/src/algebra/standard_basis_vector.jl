"""
    AutoregressiveMessagePassingRules.StandardBasisVector(length::Int, index::Int, scale = 1)

The vector of `length` entries that is zero everywhere except for `scale` at `index`, the
standard basis vector `scale ⋅ eᵢ`, held as those three numbers. The AR rules use `e₁` to pick
the first component of the state; a model can use one to pick an entry of a vector, as in
`softdot(x, StandardBasisVector(n, j), γ)` with the SoftDot node, in place of a dense one-hot
vector, which gives the same messages more slowly: the products the rules form read one entry
instead of `length`.

It is an `AbstractVector`, so it works wherever a vector does. The products written out without
building the dense vector are those with a number, `dot` with a dense `Vector` or
another basis vector, `dot(e₁, A, e₂)` and `A * e` with a dense `Matrix`, `v * eᵀ`, and
[`scaled_outer`](@extref MessagePassingRulesBase.scaled_outer)`(e, a)`. Any other operation
reads its entries.

# Throws

- `ArgumentError` when `length < 1` or `index` is not in `1:length`.

# Examples

```jldoctest; setup = :(using AutoregressiveMessagePassingRules)
julia> e = AutoregressiveMessagePassingRules.StandardBasisVector(3, 2)
3-element AutoregressiveMessagePassingRules.StandardBasisVector{Int64}:
 0
 1
 0

julia> using LinearAlgebra: dot

julia> dot(e, [10.0, 20.0, 30.0])
20.0
```
"""
struct StandardBasisVector{T <: Real} <: AbstractVector{T}
    length::Int
    index::Int
    scale::T

    function StandardBasisVector(length::Int, index::Int, scale::T = 1) where {T <: Real}
        (length >= 1 && 1 <= index <= length) ||
            throw(ArgumentError("a standard basis vector of length $length has no index $index"))
        return new{T}(length, index, scale)
    end
end

Base.size(e::StandardBasisVector) = (e.length,)

Base.@propagate_inbounds function Base.getindex(e::StandardBasisVector, i::Int)
    @boundscheck checkbounds(e, i)
    return ifelse(e.index == i, e.scale, zero(e.scale))
end

# e₁ of the AR state, as a standard basis vector in `T`, or `one(T)` for a univariate AR(1).
ar_unit(::Type{V}, order) where {V <: VariateForm} = ar_unit(Float64, V, order)
ar_unit(::Type{T}, ::Type{Multivariate}, order) where {T <: Real} = StandardBasisVector(order, 1, one(T))
ar_unit(::Type{T}, ::Type{Univariate}, order) where {T <: Real} = one(T)

function check_same_length(a, b)
    length(a) == length(b) || throw(DimensionMismatch("vectors have lengths $(length(a)) and $(length(b))"))
    return nothing
end

const AdjointBasisVector{T} = LinearAlgebra.Adjoint{T, StandardBasisVector{T}}

Base.:*(e::StandardBasisVector, x::Real) = StandardBasisVector(length(e), e.index, e.scale * x)
Base.:*(x::Real, e::StandardBasisVector) = StandardBasisVector(length(e), e.index, x * e.scale)

LinearAlgebra.dot(e::StandardBasisVector, v::Vector{<:Real}) = (check_same_length(e, v); e.scale * v[e.index])
LinearAlgebra.dot(v::Vector{<:Real}, e::StandardBasisVector) = (check_same_length(v, e); v[e.index] * e.scale)

function LinearAlgebra.dot(e1::StandardBasisVector, e2::StandardBasisVector)
    check_same_length(e1, e2)
    T = promote_type(eltype(e1), eltype(e2))
    return e1.index == e2.index ? convert(T, e1.scale * e2.scale) : zero(T)
end

function LinearAlgebra.dot(e1::StandardBasisVector, A::Matrix{<:Real}, e2::StandardBasisVector)
    size(A) == (length(e1), length(e2)) ||
        throw(DimensionMismatch("a matrix of size $(size(A)) between vectors of lengths $(length(e1)) and $(length(e2))"))
    return e1.scale * A[e1.index, e2.index] * e2.scale
end

# A e: the column of A at the index, scaled.
function Base.:*(A::Matrix{<:Real}, e::StandardBasisVector)
    size(A, 2) == length(e) || throw(DimensionMismatch("cannot multiply a matrix of size $(size(A)) by a vector of length $(length(e))"))
    return scale!!(e.scale, A[:, e.index])
end

# e a eᵀ, the precision `dot`'s rules build from it: a diagonal with its one entry.
function MessagePassingRulesBase.scaled_outer(e::StandardBasisVector, a::Real)
    T = promote_type(eltype(e), typeof(a))
    diagonal = zeros(T, length(e))
    diagonal[e.index] = e.scale * a * e.scale
    return LinearAlgebra.Diagonal(diagonal)
end

# v eᵀ: v, scaled, in the column at the index.
function Base.:*(v::Vector{<:Real}, a::AdjointBasisVector)
    e = parent(a)
    result = zeros(promote_type(eltype(v), eltype(e)), length(v), length(e))
    @inbounds for k in eachindex(v)
        result[k, e.index] = v[k] * e.scale
    end
    return result
end
