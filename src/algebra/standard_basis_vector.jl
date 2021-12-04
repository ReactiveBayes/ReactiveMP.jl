export StandardBasisVector

import Base: *
import LinearAlgebra: dot, transpose, mul!

using LinearAlgebra

@doc raw"""
`StandardBasisVector{len, ind}()` creates a standard Cartesian basis vector of zeros of length `len` with a single one at index `ind`.

An example is the 3-dimensional standard basis vector for the first dimension
```math
e = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
```
Which can be constructed by calling `e = StandardBasisVector(3,1)`
"""
struct StandardBasisVector{N, T} <: AbstractVector{T}
    function StandardBasisVector(len::Int, ind::Int)
        @assert ind >= 0 && len >= 1 && ind <= len
        return new{len, ind}()
    end
end

# extensions of base functionality
Base.size(e::StandardBasisVector{N, T})    where { N, T } = (N,)
Base.size(e::StandardBasisVector{N, T}, d) where { N, T } = d::Integer == 1 ? N : 1
Base.length(e::StandardBasisVector{N, T})  where { N, T } = N

function Base.getindex(e::StandardBasisVector{N,T}, i::Int) where { N, T }
    @assert i > 0 && i <= N
    if T == i
        return 1
    else 
        return 0
    end
end

function Base.show(io::IO, ::MIME"text/plain", ::StandardBasisVector{N,T}) where { N, T }
    if N < 10
        x = zeros(N)
        x[T] = 1
        print(io, x)
    else
        if T == 1
            print(io, "[1, 0, 0, ...]")
        elseif T == 2
            print(io, "[0, 1, 0, ...]")
        else
            print(io, "[..., 0, 1, 0, ...]\n         ^\n         $T")
        end
    end
end
function Base.show(io::IO, ::StandardBasisVector{N,T}) where { N, T }
    if N < 10
        x = zeros(N)
        x[T] = 1
        print(io, x)
    else
        compact = get(io, :compact, false)
        if T == 1
            print(io, "[1, 0, 0, ...]")
        elseif T == 2
            print(io, "[0, 1, 0, ...]")
        else
            if compact
                print(io, "[..., 0, 1, 0, ...]\n         ^\n         $T")
            else
                print(io, "[0, ..., 1 at index $T, ..., 0]")
            end
        end
    end
end

# get index function
getind(e::StandardBasisVector{N, T}) where { N, T }= T
LinearAlgebra.adjoint(e::StandardBasisVector{N, T}) where { N, T } = Adjoint{Int64, StandardBasisVector{N, T}}(e)

# standard basis vector - scalar
function Base.:*(e::StandardBasisVector{N,T}, x::T1) where { N, T, T1 <: Real}
    y = zeros(T1, N)
    y[T] = x
    return y
end

function Base.:*(x::T1, e::StandardBasisVector{N,T}) where { N, T, T1 <: Real}
    y = zeros(T1, N)
    y[T] = x
    return y
end

function Base.:*(::Adjoint{T2, StandardBasisVector{N,T}}, x::T1) where { N, T, T1 <: Real, T2}
    y = zeros(T1, 1, N)
    y[T] = x
    return y
end

function Base.:*(x::T1, ::Adjoint{T2, StandardBasisVector{N,T}}) where { N, T, T1 <: Real, T2 }
    y = zeros(T1, 1, N)
    y[T] = x
    return y
end

# dot product
function LinearAlgebra.dot(::StandardBasisVector{N,T}, v::AbstractVector) where { N, T }
    @assert length(v) == N
    return v[T]
end

function LinearAlgebra.dot(v::AbstractVector, ::StandardBasisVector{N,T}) where { N, T }
    @assert length(v) == N
    return v[T]
end

function LinearAlgebra.dot(::StandardBasisVector{N1,T1}, ::StandardBasisVector{N2,T2}) where { N1, N2, T1, T2}
    @assert N1 == N2
    return T1 == T2 ? 1 : 0
end

function LinearAlgebra.dot(::StandardBasisVector{N1,T1}, A::AbstractMatrix, ::StandardBasisVector{N2,T2}) where { N1, N2, T1, T2 }
    @assert size(A) == (N1, N2)
    return A[T1, T2]
    
end

# vector - vector
function Base.:*(v::AbstractVector{T2}, ::Adjoint{T1, StandardBasisVector{N,T}}) where { N, T, T1, T2 }
    lv = length(v)
    result = zeros(T2, lv, N)
    @inbounds @simd for k in 1:lv
        result[k, T] = v[k]
    end
    return result
end

function Base.:*(::StandardBasisVector{N,T}, v::Adjoint{T1, <: AbstractVector{T1}}) where { N, T, T1 } 
    lv = length(v)
    result = zeros(T1, N, lv)
    @inbounds @simd for k in 1:lv
        result[T, k] = v[k]
    end
    return result
end

# vector matrix
function Base.:*(A::AbstractMatrix, ::StandardBasisVector{N,T}) where { N, T }
    @assert size(A,2) == N
    return A[:, T]
end

function Base.:*(A::Adjoint{T1, <: AbstractMatrix{T1}}, ::StandardBasisVector{N,T}) where { N, T, T1 }
    @assert size(A,2) == N
    return A[:,T]
end

function Base.:*(A::AbstractMatrix{T2}, ::Adjoint{T1, StandardBasisVector{N,T}}) where { N, T, T1, T2 }
    sA = size(A)
    @assert sA[2] == 1
    result = zeros(T2, sA[1], N)
    @inbounds @simd for k in 1:sA[1]
        result[k, T] = A[k]
    end
    return result
end

function Base.:*(A::Adjoint{T1, <: AbstractMatrix{T1}}, ::Adjoint{T2, StandardBasisVector{N,T}}) where { N, T, T1, T2 }
    sA = size(A)
    @assert sA[2] == 1
    result = zeros(T1, sA[1], N)
    @inbounds @simd for k in 1:sA[1]
        result[k, T] = A[k]
    end
    return result
end

function Base.:*(::StandardBasisVector{N,T}, A::AbstractMatrix{T1}) where { N, T, T1 }
    sA = size(A)
    @assert sA[1] == 1
    result = zeros(T1, N, sA[2])
    @inbounds @simd for k in 1:sA[2]
        result[T,k] = A[k]
    end
    return result
end

function Base.:*(::StandardBasisVector{N,T}, A::Adjoint{T1, <: AbstractMatrix{T1}}) where { N, T, T1 }
    @assert size(A,2) == N
    return A[:,T]
end

# custom
function v_a_vT(e1::StandardBasisVector{N1,T1}, a::T3, e2::StandardBasisVector{N2, T2}) where { N1, N2, T1, T2, T3 <: Real }

    Y = zeros(T3, N1, N2)
    Y[T1,T2] = a

    # return output 
    return Y
    
end

function v_a_vT(e::StandardBasisVector{N,T}, a::T1) where { N, T, T1 <: Real }

    Y = zeros(T1, N, N)
    Y[T,T] = a

    # return output 
    return Y
    
end