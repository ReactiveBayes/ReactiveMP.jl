export diageye

import LinearAlgebra
import LinearAlgebra: Diagonal

diageye(::Type{T}, n::Int) where { T <: Real } = Matrix{T}(I, n, n)
diageye(n::Int)                                = diageye(Float64, n)

negate_inplace!(A::AbstractArray) = map!(-, A, A)
negate_inplace!(A::Real)          = -A

mul_inplace!(alpha, A::AbstractArray) = lmul!(alpha, A)
mul_inplace!(alpha, A::Real)          = alpha * A

"""
    rank1update(A, x)
    rank1update(A, x, y)

Helper function for A + x * y'. Uses optimised BLAS version for AbstractFloats and fallbacks to a generic implementation in case of differentiation
"""
rank1update(A::AbstractMatrix, x::AbstractVector)                    = rank1update(eltype(A), eltype(x), eltype(x), A, x, x)
rank1update(A::AbstractMatrix, x::AbstractVector, y::AbstractVector) = rank1update(eltype(A), eltype(x), eltype(y), A, x, y)

rank1update(A::Real, x::Real)          = rank1update(A, x, x)
rank1update(A::Real, x::Real, y::Real) = A + x * y

function rank1update(::Type{ T }, ::Type{ T }, ::Type{T}, A::AbstractMatrix, x::AbstractVector, y::AbstractVector) where { T <: AbstractFloat } 
    return LinearAlgebra.BLAS.ger!(one(T), x, y, copy(A))
end

function rank1update(::Type{ T1 }, ::Type{ T2 }, ::Type{ T3 }, A::AbstractMatrix, x::AbstractVector, y::AbstractVector) where { T1 <: Real, T2 <: Real , T3 <: Real } 
    return A + x * y'
end


"""
    mul_trace(A, B)

Computes tr(A * B) wihtout allocating A * B.
"""
function mul_trace end

mul_trace(A::Real, B::Real) = A * B

function mul_trace(A::AbstractMatrix, B::AbstractMatrix)
    sA, sB = size(A), size(B)
    @assert (sA === sB) && (length(sA) === 2) && (first(sA) === last(sA))
    result = zero(promote_type(eltype(A), eltype(B)))
    @turbo for i in 1:first(sA), j in 1:first(sA)
        @inbounds result += A[i, j] * B[j, i]
    end
    return result
end