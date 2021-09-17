export PermutationMatrix, PT_X_P, PT_X_P!, getind ,getindices

import Base: *
import LinearAlgebra: transpose , inv, mul!

@doc raw"""
`PermutationMatrix(ind::Array{T})` creates a permutation matrix with ones at coordinates `(k, ind[k]) for k = 1:length(ind)`.

A permutation matrix represents a matrix containing only zeros and ones, which basically permutes the vector or matrix it is multiplied with.
These matrices A are constrained by:
```math
    A_{ij} = \{0, 1\}\\
    ∑_{i} A_{ij} = 1\\
    ∑_{j} A_{ij} = 1
```
An example is the 3-dimensional permutation matrix
```math
A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{bmatrix}
```
"""
struct PermutationMatrix{ T <: Int } <: AbstractMatrix{T}
    ind::Vector{T}
    function PermutationMatrix(ind::Vector{T}) where { T <: Int }
        return new{T}(ind)
    end
end

@doc raw"""
`PermutationMatrix(dim::T)` Generates a random permutation matrix of size (dim x dim).
"""
function PermutationMatrix(dim::T) where { T <: Int }
    ind = shuffle(collect(1:dim))
    return PermutationMatrix(ind)
end


# extensions of base functionality
Base.eltype(::PermutationMatrix{T}) where { T } = T
Base.size(mat::PermutationMatrix)               = (length(mat), length(mat)) 
Base.length(mat::PermutationMatrix)             = length(mat.ind)

function Base.getindex(mat::PermutationMatrix, i::Int, j::Int)
    if mat.ind[i] == j
        return one(eltype(mat))
    else 
        return zero(eltype(mat))
    end
end

LinearAlgebra.inv(mat::PermutationMatrix)       = adjoint(mat)


# get functions
getind(mat::PermutationMatrix)      = mat.ind
getindices(mat::PermutationMatrix)  = mat.ind
getind(mat::Adjoint{T, PermutationMatrix{T}}) where { T }       = sortperm(mat.parent.ind)
getindices(mat::Adjoint{T, PermutationMatrix{T}}) where { T }   = sortperm(mat.parent.ind)


# Permutation-vector multiplication
function Base.:*(P::PermutationMatrix, v::AbstractVector)
    y = similar(v)
    mul!(y, P, v)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, P::PermutationMatrix, v::AbstractVector)
    ind = getind(P)
    @inbounds @simd for k in 1:length(P)
        y[k] = v[ind[k]]
    end
end

function Base.:*(P::Adjoint{T, PermutationMatrix{T}}, v::AbstractVector) where { T }
    y = similar(v)
    mul!(y, P, v)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, P::Adjoint{T, PermutationMatrix{T}}, v::AbstractVector) where { T }
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:length(P)
        y[ind[k]] = v[k]
    end
end


function Base.:*(P::Transpose{T, PermutationMatrix{T}}, v::AbstractVector) where { T }
    y = similar(v)
    mul!(y, P, v)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, P::Transpose{T, PermutationMatrix{T}}, v::AbstractVector) where { T }
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:length(P)
        y[ind[k]] = v[k]
    end
end


# Permutation-matrix multiplication
function Base.:*(P::PermutationMatrix, X::AbstractMatrix)
    @assert size(X,1) == size(X,2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, P, X)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, P::PermutationMatrix{T}, X::AbstractMatrix) where { T }
    ind = getind(P)
    @inbounds @simd for k in 1:length(P)
        @inbounds @simd for ki = 1:length(P)
            Y[k,ki] = X[ind[k],ki]
        end
    end
end

function Base.:*(P::Adjoint{T, PermutationMatrix{T}}, X::AbstractMatrix) where { T }
    @assert size(X,1) == size(X,2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, P, X)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, P::Adjoint{T, PermutationMatrix{T}}, X::AbstractMatrix) where { T }
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:length(P)
        @inbounds @simd for ki = 1:length(P)
            Y[ind[k],ki] = X[k,ki]
        end
    end
end

function Base.:*(P::Transpose{T, PermutationMatrix{T}}, X::AbstractMatrix) where { T }
    @assert size(X,1) == size(X,2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, P, X)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, P::Transpose{T, PermutationMatrix{T}}, X::AbstractMatrix) where { T }
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:length(P)
        @inbounds @simd for ki = 1:length(P)
            Y[ind[k],ki] = X[k,ki]
        end
    end
end

# transposedvector - matrix product
Base.:*(x::Adjoint{T, S}, P::PermutationMatrix) where { T, S <: AbstractVector } = (P'*x')'
Base.:*(x::Adjoint{T, S}, P::Adjoint{T1, PermutationMatrix{T1}}) where { T, T1, S <: AbstractVector } = (P'*x')'
Base.:*(x::Adjoint{T, S}, P::Transpose{T1, PermutationMatrix{T1}}) where { T, T1, S <: AbstractVector } = (P'*x')'
Base.:*(x::Transpose{T, S}, P::PermutationMatrix) where { T, S <: AbstractVector } = transpose(transpose(P)*transpose(x))
Base.:*(x::Transpose{T, S}, P::Adjoint{T1, PermutationMatrix{T1}}) where { T, T1, S <: AbstractVector } = transpose(transpose(P)*transpose(x))
Base.:*(x::Transpose{T, S}, P::Transpose{T1, PermutationMatrix{T1}}) where { T, T1, S <: AbstractVector } = transpose(transpose(P)*transpose(x))


# matrix-Permutation multiplication
function Base.:*(X::AbstractMatrix, P::PermutationMatrix)
    @assert size(X,1) == size(X,2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, X, P)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, P::PermutationMatrix)
    ind = getind(P)
    @inbounds @simd for k in 1:length(P)
        @inbounds @simd for ki = 1:length(P)
            Y[ki,ind[k]] = X[ki,k]
        end
    end
end

function Base.:*(X::AbstractMatrix, P::Adjoint{T, PermutationMatrix{T}}) where { T }
    @assert size(X,1) == size(X,2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, X, P)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, P::Adjoint{T, PermutationMatrix{T}}) where { T }
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:length(P)
        @inbounds @simd for ki = 1:length(P)
            Y[ki,k] = X[ki,ind[k]]
        end
    end
end

function Base.:*(X::AbstractMatrix, P::Transpose{T, PermutationMatrix{T}}, ) where { T }
    @assert size(X,1) == size(X,2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, X, P)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, P::Transpose{T, PermutationMatrix{T}}, ) where { T }
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:length(P)
        @inbounds @simd for ki = 1:length(P)
            Y[ki,k] = X[ki,ind[k]]
        end
    end
end


# multiplication of some square matrix with the permutation matrix 
function PT_X_P(X::AbstractMatrix, P::PermutationMatrix)

    # allocate output
    Y = copy(X)

    # perform permutation operation
    PT_X_P!(Y, X, P)

    # return output 
    return Y
    
end

function PT_X_P!(Y::AbstractMatrix, X::AbstractMatrix, P::PermutationMatrix)

    # fetch indices of permutation matrix
    ind = getind(P)

    # perform permutation operation
    @inbounds @simd for k1 = 1:length(P)
        @inbounds @simd for k2 = 1:length(P)
            Y[ind[k1], ind[k2]] = X[k1,k2] 
        end
    end
    
end