export diageye

using StatsFuns: logistic
using StatsFuns: softmax, softmax!

import LinearAlgebra
import LoopVectorization: @tturbo

diageye(::Type{T}, n::Int) where { T <: Real } = Matrix{T}(I, n, n)
diageye(n::Int)                                = diageye(Float64, n)

function normalize_sum(x::Array{Float64,1}) 
    x ./ sum(x)
end

const sigmoid = logistic

dtanh(x) = 1 - tanh(x)^2
"""
    negate_inplace!(A)

Returns `-A`, modifying and reusing `A` storage if possible.

See also: [`mul_inplace!`](@ref)
"""
function negate_inplace! end

negate_inplace!(A::AbstractArray) = -A
negate_inplace!(A::Array)         = map!(-, A, A)
negate_inplace!(A::Real)          = -A

"""
    mul_inplace!(alpha, A)

Returns `alpha * A`, modifying and reusing `A` storage if possible.

See also: [`negate_inplace!`](@ref)
"""
function mul_inplace! end

mul_inplace!(alpha, A::AbstractArray) = alpha * A
mul_inplace!(alpha, A::Array)         = lmul!(alpha, A)
mul_inplace!(alpha, A::Real)          = alpha * A

"""
    rank1update(A, x)
    rank1update(A, x, y)

Helper function for A + x * y'. Uses optimised BLAS version for AbstractFloats and fallbacks to a generic implementation in case of differentiation
"""
function rank1update end

rank1update(A::AbstractMatrix, x::AbstractVector)                    = rank1update(eltype(A), eltype(x), eltype(x), A, x, x)
rank1update(A::AbstractMatrix, x::AbstractVector, y::AbstractVector) = rank1update(eltype(A), eltype(x), eltype(y), A, x, y)

rank1update(A::Real, x::Real)          = rank1update(A, x, x)
rank1update(A::Real, x::Real, y::Real) = A + x * y

function rank1update(::Type{ T }, ::Type{ T }, ::Type{T}, A::Matrix, x::Vector, y::Vector) where { T <: AbstractFloat } 
    return LinearAlgebra.BLAS.ger!(one(T), x, y, copy(A))
end

function rank1update(::Type{ T1 }, ::Type{ T2 }, ::Type{ T3 }, A::AbstractMatrix, x::AbstractVector, y::AbstractVector) where { T1 <: Real, T2 <: Real , T3 <: Real } 
    return A + x * y'
end


"""
    mul_trace(A, B)

Computes tr(A * B) without allocating A * B.
"""
function mul_trace end

mul_trace(A::Real, B::Real) = A * B

function mul_trace(A::AbstractMatrix, B::AbstractMatrix)
    sA, sB = size(A), size(B)
    @assert (sA === sB) && (length(sA) === 2) && (first(sA) === last(sA))
    result = zero(promote_type(eltype(A), eltype(B)))
    @tturbo for i in 1:first(sA), j in 1:first(sA)
        @inbounds result += A[i, j] * B[j, i]
    end
    return result
end


"""
    v_a_vT(v, a)

Computes v*a*v^T with a single allocation.
"""
function v_a_vT(v::AbstractVector, a::T) where { T <: Real }
    result = v*v'
    result .*= a
    return result
end

"""
    v_a_vT(v1, a, v2)

Computes v1*a*v2^T with a single allocation.
"""
function v_a_vT(v1::AbstractVector, a::T, v2::AbstractVector) where { T <: Real }
    result = v1*v2'
    result .*= a
    return result
end