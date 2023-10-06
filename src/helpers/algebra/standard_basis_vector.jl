export StandardBasisVector

import Base: *
import LinearAlgebra: dot, transpose, mul!

using LinearAlgebra

@doc raw"""

    StandardBasisVector{T, len, ind}(scale::T)

`StandardBasisVector` creates a standard Cartesian basis vector of zeros of length `len` with a single element `scale` at index `ind`.

An example is the 3-dimensional standard basis vector for the first dimension
```math
e = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
```
Which can be constructed by calling `e = StandardBasisVector(3, 1, 1)`
"""
struct StandardBasisVector{T} <: AbstractVector{T}
    length :: Int
    index  :: Int
    scale  :: T
end

function StandardBasisVector(length::Int, index::Int, scale::T = one(Int)) where {T <: Real}
    @assert length >= 0 && (1 <= index <= length)
    return StandardBasisVector{T}(length, index, scale)
end

Base.eltype(::StandardBasisVector{T}) where {T}       = T
Base.eltype(::Type{StandardBasisVector{T}}) where {T} = T

# extensions of base functionality
Base.size(e::StandardBasisVector)    = (length(e),)
Base.size(e::StandardBasisVector, d) = d::Integer == 1 ? length(e) : 1
Base.length(e::StandardBasisVector)  = e.length

Base.@propagate_inbounds function Base.getindex(e::StandardBasisVector, i::Int)
    @boundscheck checkbounds(e, i)
    return ifelse(getind(e) === i, e.scale, zero(eltype(e)))
end

function Base.show(io::IO, ::MIME"text/plain", e::StandardBasisVector{T}) where {T}
    N = length(e)
    I = getind(e)
    if N < 10
        x = zeros(T, N)
        x[I] = e.scale
        print(io, x)
    else
        if I == 1
            print(io, "[$(e.scale), 0, 0, ...]")
        elseif I == 2
            print(io, "[0, $(e.scale), 0, ...]")
        else
            print(io, "[..., 0, $(e.scale), 0, ...]\n         ^\n         $I")
        end
    end
end

function Base.show(io::IO, e::StandardBasisVector{T}) where {T}
    N = length(e)
    I = getind(e)
    if N < 10
        x = zeros(T, N)
        x[I] = e.scale
        print(io, x)
    else
        compact = get(io, :compact, false)
        if I == 1
            print(io, "[$(e.scale), 0, 0, ...]")
        elseif I == 2
            print(io, "[0, $(e.scale), 0, ...]")
        else
            if compact
                print(io, "[..., 0, $(e.scale), 0, ...]\n         ^\n         $I")
            else
                print(io, "[0, ..., $(e.scale) at index $I, ..., 0]")
            end
        end
    end
end

# get index function
getind(e::StandardBasisVector) = e.index

LinearAlgebra.adjoint(e::S) where {S <: StandardBasisVector} = Adjoint{eltype(S), S}(e)

# standard basis vector - scalar
Base.:*(e::StandardBasisVector, x::Real) = StandardBasisVector(length(e), getind(e), e.scale * x)
Base.:*(x::Real, e::StandardBasisVector) = StandardBasisVector(length(e), getind(e), x * e.scale)

Base.:*(a::Adjoint{T, StandardBasisVector{T}}, x::Real) where {T} = (a' * x)'
Base.:*(x::Real, a::Adjoint{T, StandardBasisVector{T}}) where {T} = (x * a')'

# dot product
function LinearAlgebra.dot(e::StandardBasisVector, v::AbstractVector)
    @assert length(v) === length(e)
    return e.scale * v[getind(e)]
end

function LinearAlgebra.dot(v::AbstractVector, e::StandardBasisVector)
    @assert length(v) === length(e)
    return v[getind(e)] * e.scale
end

function LinearAlgebra.dot(e1::StandardBasisVector{T1}, e2::StandardBasisVector{T2}) where {T1, T2}
    @assert length(e1) === length(e2)
    T = promote_type(T1, T2)
    return ifelse(getind(e1) === getind(e2), convert(T, e1.scale * e2.scale), zero(T))::T
end

@inline function __dot3_basis_vector_mat(e1, A, e2)
    @assert size(A) == (length(e1), length(e2))
    return e1.scale * A[getind(e1), getind(e2)] * e2.scale
end

# Julia does not understand union here and throws an ambiguity error
LinearAlgebra.dot(e1::StandardBasisVector, A::AbstractMatrix, e2::StandardBasisVector) = __dot3_basis_vector_mat(e1, A, e2)
LinearAlgebra.dot(e1::StandardBasisVector, A::Diagonal, e2::StandardBasisVector) = __dot3_basis_vector_mat(e1, A, e2)
LinearAlgebra.dot(e1::StandardBasisVector, A::Adjoint{T, <:AbstractMatrix{T}}, e2::StandardBasisVector) where {T} = __dot3_basis_vector_mat(e1, A, e2)

# vector - vector
function Base.:*(v::AbstractVector{T1}, a::Adjoint{T2, StandardBasisVector{T2}}) where {T1 <: Real, T2 <: Real}
    parent = a'
    N = length(parent)
    I = getind(parent)
    T = promote_type(T1, T2)
    lv = length(v)
    s = parent.scale
    result = zeros(T, lv, N)
    @inbounds @simd for k in 1:lv
        result[k, I] = v[k] * s
    end
    return result
end

function Base.:*(v::StandardBasisVector{T1}, a::Adjoint{T2, StandardBasisVector{T2}}) where {T1 <: Real, T2 <: Real}
    T = promote_type(T1, T2)
    N1 = length(v)
    I1 = getind(v)
    p2 = a'
    N2 = length(p2)
    I2 = getind(p2)
    s = p2.scale
    result = zeros(T, N1, N2)
    result[I1, I2] = v.scale * s
    return result
end

function Base.:*(e::StandardBasisVector{T1}, a::Adjoint{T2, <:AbstractVector{T2}}) where {T1 <: Real, T2 <: Real}
    N = length(e)
    I = getind(e)
    T = promote_type(T1, T2)
    lv = length(a)
    s = e.scale
    result = zeros(T, N, lv)
    @inbounds @simd for k in 1:lv
        result[I, k] = s * a[k]
    end
    return result
end

function Base.:*(v::Adjoint{T1, <:AbstractVector{T1}}, e::StandardBasisVector{T2}) where {T1 <: Real, T2 <: Real}
    @assert length(v) === length(e)
    return v[getind(e)] * e.scale
end

# vector matrix

@inline function __mul_mat_basis_vector(A, e)
    @assert size(A, 2) === length(e)
    v = A[:, getind(e)]
    v = mul_inplace!(e.scale, v)
    return v
end

# Julia does not understand `Union` here and throws an ambiguity error
Base.:*(A::AbstractMatrix, e::StandardBasisVector) = __mul_mat_basis_vector(A, e)
Base.:*(A::Diagonal, e::StandardBasisVector) = __mul_mat_basis_vector(A, e)
Base.:*(A::Adjoint{T, <:AbstractMatrix{T}}, e::StandardBasisVector) where {T <: Real} = __mul_mat_basis_vector(A, e)

@inline function __mul_mat_adjoint_basis_vector(A, e)
    sA = size(A)
    @assert sA[2] === 1
    p      = e'
    N      = length(p)
    I      = getind(p)
    T      = promote_type(eltype(A), eltype(e))
    s      = p.scale
    result = zeros(T, sA[1], N)
    @inbounds @simd for k in 1:sA[1]
        result[k, I] = A[k] * s
    end
    return result
end

Base.:*(A::AbstractMatrix, e::Adjoint{T2, StandardBasisVector{T2}}) where {T2} = __mul_mat_adjoint_basis_vector(A, e)
Base.:*(A::Diagonal, e::Adjoint{T2, StandardBasisVector{T2}}) where {T2} = __mul_mat_adjoint_basis_vector(A, e)

@inline function __mul_basis_vector_mat(e, A)
    sA = size(A)
    @assert sA[1] === 1
    N      = length(e)
    I      = getind(e)
    T      = promote_type(eltype(e), eltype(A))
    s      = e.scale
    result = zeros(T, N, sA[2])
    @inbounds @simd for k in 1:sA[2]
        result[I, k] = s * A[k]
    end
    return result
end

Base.:*(e::StandardBasisVector, A::AbstractMatrix) = __mul_basis_vector_mat(e, A)
Base.:*(e::StandardBasisVector, A::Diagonal) = __mul_basis_vector_mat(e, A)

function Base.:*(e::StandardBasisVector, A::Adjoint{T, <:AbstractMatrix{T}}) where {T <: Real}
    @assert size(A, 2) === length(e)
    v = A[:, getind(e)]
    v = mul_inplace!(e.scale, v)
    return v
end

# custom
function v_a_vT(e1::StandardBasisVector{T1}, a::T3, e2::StandardBasisVector{T2}) where {T1 <: Real, T2 <: Real, T3 <: Real}
    T = promote_type(T1, T3, T2)
    Y = zeros(T, length(e1), length(e2))
    Y[getind(e1), getind(e2)] = e1.scale * a * e2.scale

    # return output 
    return Y
end

function v_a_vT(e::StandardBasisVector{T1}, a::T2) where {T1 <: Real, T2 <: Real}
    N = length(e)
    I = getind(e)
    T = promote_type(T1, T2)
    Y = zeros(T, N)
    Y[I] = e.scale * a * e.scale
    # return output 
    return Diagonal(Y)
end
