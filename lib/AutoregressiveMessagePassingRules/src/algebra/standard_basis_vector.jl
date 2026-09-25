# The vector of length `length` that is zero except for `scale` at `index`: `ar_unit`'s e₁, which
# picks the first component of the AR state. Internal. Its products are written out only
# against a scalar, a dense `Matrix` or `Vector` and each other, the combinations the rules
# form; any other operand takes the generic `AbstractVector` fallback, which reads `getindex`.
# v6 wrote them against `AbstractMatrix`, `Diagonal` and `Adjoint`, which Aqua counted at 85
# ambiguities.
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
    return StandardMessagePassingRules.mul_inplace!(e.scale, A[:, e.index])
end

# e a eᵀ, the precision `dot`'s rules build from it: a diagonal with its one entry, as in v6.
function StandardMessagePassingRules.v_a_vT(e::StandardBasisVector, a::Real)
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
