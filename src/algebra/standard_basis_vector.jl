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
struct StandardBasisVector{N, I, T} <: AbstractVector{T} 
    scale :: T
end

function StandardBasisVector(N::Int, I::Int, scale::T = one(Int)) where { T <: Real }
    @assert N >= 0 && (1 <= I <= N)
    return StandardBasisVector{N, I, T}(scale)
end

Base.eltype(::StandardBasisVector{N, I, T})         where { N, I, T } = T
Base.eltype(::Type{ StandardBasisVector{N, I, T} }) where { N, I, T } = T

# extensions of base functionality
Base.size(::StandardBasisVector{N})    where { N } = (N,)
Base.size(::StandardBasisVector{N}, d) where { N } = d::Integer == 1 ? N : 1
Base.length(::StandardBasisVector{N})  where { N } = N

Base.@propagate_inbounds function Base.getindex(e::StandardBasisVector{N, I}, i::Int) where { N, I }
    @boundscheck checkbounds(e, i)
    return ifelse(I === i, e.scale, zero(eltype(e)))
end

function Base.show(io::IO, ::MIME"text/plain", e::StandardBasisVector{N, I, T}) where { N, I, T }
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

function Base.show(io::IO, e::StandardBasisVector{N, I, T}) where { N, I, T }
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
getind(::StandardBasisVector{N, I}) where { N, I } = I

LinearAlgebra.adjoint(e::S) where { S <: StandardBasisVector } = Adjoint{eltype(S), S}(e)

# standard basis vector - scalar
Base.:*(e::StandardBasisVector{N, I}, x::Real) where { N, I } = StandardBasisVector(N, I, e.scale * x)
Base.:*(x::Real, e::StandardBasisVector{N, I}) where { N, I } = StandardBasisVector(N, I, x * e.scale)

Base.:*(a::Adjoint{T, StandardBasisVector{N, I, T}}, x::Real) where { N, I, T } = (a' * x)'
Base.:*(x::Real, a::Adjoint{T, StandardBasisVector{N, I, T}}) where { N, I, T } = (x * a')'

# dot product
function LinearAlgebra.dot(e::StandardBasisVector{N, I}, v::AbstractVector) where { N, I }
    @assert length(v) === N
    return e.scale * v[I]
end

function LinearAlgebra.dot(v::AbstractVector, e::StandardBasisVector{N, I}) where { N, I }
    @assert length(v) === N
    return v[I] * e.scale
end

function LinearAlgebra.dot(e1::StandardBasisVector{N, I, T1}, e2::StandardBasisVector{N, I, T2}) where { N, I, T1, T2 } 
    return e1.scale * e2.scale
end

function LinearAlgebra.dot(e1::StandardBasisVector{N, I1, T1}, e2::StandardBasisVector{N, I2, T2}) where { N, I1, I2, T1, T2 } 
    return zero(promote_type(T1, T2))
end

function LinearAlgebra.dot(e1::StandardBasisVector{N1, I1, T1}, e2::StandardBasisVector{N2, I2, T2}) where { N1, N2, I1, I2, T1, T2 } 
    error("Incompatible length of standard basis vectors in dot function. length(e1) = $(length(e1)), length(e1) = $(length(e2))")
end

function LinearAlgebra.dot(e1::StandardBasisVector{N1, I1}, A::AbstractMatrix, e2::StandardBasisVector{N2, I2}) where { N1, N2, I1, I2 }
    @assert size(A) == (N1, N2)
    return e1.scale * A[I1, I2] * e2.scale
end

# vector - vector
function Base.:*(v::AbstractVector{T1}, a::Adjoint{T2, StandardBasisVector{N, I, T2}}) where { N, I, T1 <: Real, T2 <: Real }
    T  = promote_type(T1, T2)
    lv = length(v)
    s  = (a').scale
    result = zeros(T, lv, N)
    @inbounds @simd for k in 1:lv
        result[k, I] = v[k] * s
    end
    return result
end

function Base.:*(v::StandardBasisVector{N, I1, T1}, a::Adjoint{T2, StandardBasisVector{N, I2, T2}}) where { N, I1, I2, T1 <: Real, T2 <: Real }
    T  = promote_type(T1, T2)
    lv = length(v)
    s  = (a').scale
    result = zeros(T, lv, N)
    result[I1, I2] = v.scale * s
    return result
end

function Base.:*(e::StandardBasisVector{N, I, T1}, a::Adjoint{T2, <: AbstractVector{T2}}) where { N, I, T1 <: Real, T2 <: Real } 
    T  = promote_type(T1, T2)
    lv = length(a)
    s  = e.scale
    result = zeros(T, N, lv)
    @inbounds @simd for k in 1:lv
        result[I, k] = s * a[k]
    end
    return result
end

function Base.:*(v::Adjoint{T1, <:AbstractVector{T1}}, e::StandardBasisVector{N, I, T2}) where { N, I, T1 <: Real, T2 <: Real }
    @assert length(v) === length(e)
    return v[I] * e.scale
end

# vector matrix
function Base.:*(A::AbstractMatrix, e::StandardBasisVector{N,I}) where { N, I }
    @assert size(A, 2) === N
    v = A[:, I]
    mul_inplace!(e.scale, v)
    return v
end

function Base.:*(A::Adjoint{T, <:AbstractMatrix{T}}, e::StandardBasisVector{N,I}) where { N, I, T <: Real }
    @assert size(A, 2) === N
    v = A[:, I]
    mul_inplace!(e.scale, v)
    return v
end

function Base.:*(A::AbstractMatrix{T1}, a::Adjoint{T2, StandardBasisVector{N, I, T2}}) where { N, I, T2 <: Real, T1 <: Real }
    sA = size(A)
    @assert sA[2] === 1
    T      = promote_type(T1, T2)
    s      = (a').scale
    result = zeros(T, sA[1], N)
    @inbounds @simd for k in 1:sA[1]
        result[k, I] = A[k] * s
    end
    return result
end


function Base.:*(e::StandardBasisVector{N, I, T1}, A::AbstractMatrix{T2}) where { N, I, T1 <: Real, T2 <: Real }
    sA = size(A)
    @assert sA[1] === 1
    T      = promote_type(T1, T2)
    s      = e.scale
    result = zeros(T, N, sA[2])
    @inbounds @simd for k in 1:sA[2]
        result[I, k] = s * A[k]
    end
    return result
end

function Base.:*(e::StandardBasisVector{N,T}, A::Adjoint{T1, <: AbstractMatrix{T1}}) where { N, T <: Real, T1 <: Real }
    @assert size(A, 2) === N
    v = A[:, T]
    mul_inplace!(e.scale, v)
    return v
end

# custom
function v_a_vT(e1::StandardBasisVector{N1, I1, T1}, a::T3, e2::StandardBasisVector{N2, I2, T2}) where { N1, N2, I1, I2, T1 <: Real, T2 <: Real, T3 <: Real }

    T = promote_type(T1, T3, T2)
    Y = zeros(T, N1, N2)
    Y[I1, I2] = e1.scale * a * e2.scale

    # return output 
    return Y
    
end

function v_a_vT(e::StandardBasisVector{N, I, T1}, a::T2) where { N, I, T1 <: Real, T2 <: Real }

    T = promote_type(T1, T2)
    Y = zeros(T, N, N)
    Y[I, I] = e.scale * a * e.scale

    # return output 
    return Y
    
end