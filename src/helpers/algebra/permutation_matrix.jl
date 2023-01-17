export PermutationMatrix

import Base: *
import LinearAlgebra: transpose, inv, mul!

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
struct PermutationMatrix{T <: Integer} <: AbstractMatrix{T}
    ind::Vector{T}
    function PermutationMatrix(ind::Vector{T}) where {T <: Integer}
        return new{T}(ind)
    end
end

@doc raw"""
`PermutationMatrix(dim::T; switch_first::Bool=true)` Generates a random permutation matrix of size (dim x dim).
The `switch_first` argument specifies whether the first index always has to be permuted.
"""
function PermutationMatrix(dim::T; switch_first::Bool = true) where {T <: Integer}
    ind = shuffle(collect(1:dim))
    if switch_first && ind[1] == 1
        tmp = ind[2]
        ind[2] = ind[1]
        ind[1] = tmp
    end
    return PermutationMatrix(ind)
end

# extensions of base functionality
Base.eltype(::PermutationMatrix{T}) where {T} = T
function Base.size(mat::PermutationMatrix)
    nr_elements = length(mat.ind)
    return (nr_elements, nr_elements)
end
Base.size(mat::PermutationMatrix, d) = d::Integer <= 2 ? length(mat.ind) : 1
Base.length(mat::PermutationMatrix)  = prod(size(mat))

function Base.getindex(mat::PermutationMatrix, i::Int, j::Int)
    if mat.ind[i] == j
        return one(eltype(mat))
    else
        return zero(eltype(mat))
    end
end

LinearAlgebra.inv(mat::PermutationMatrix) = adjoint(mat)

# get functions
getind(mat::PermutationMatrix) = mat.ind
getind(mat::Adjoint{T, PermutationMatrix{T}}) where {T} = sortperm(mat.parent.ind)

# Permutation-vector multiplication
function Base.:*(P::PermutationMatrix, v::AbstractVector)
    y = similar(v)
    mul!(y, P, v)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, P::PermutationMatrix, v::AbstractVector)
    ind = getind(P)
    @inbounds @simd for k in 1:size(P, 1)
        y[k] = v[ind[k]]
    end
end

function Base.:*(P::Adjoint{T, PermutationMatrix{T}}, v::AbstractVector) where {T}
    y = similar(v)
    mul!(y, P, v)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, P::Adjoint{T, PermutationMatrix{T}}, v::AbstractVector) where {T}
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:size(P, 1)
        y[ind[k]] = v[k]
    end
end

function Base.:*(P::Transpose{T, PermutationMatrix{T}}, v::AbstractVector) where {T}
    y = similar(v)
    mul!(y, P, v)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, P::Transpose{T, PermutationMatrix{T}}, v::AbstractVector) where {T}
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:size(P, 1)
        y[ind[k]] = v[k]
    end
end

# Permutation-matrix multiplication
function Base.:*(P::PermutationMatrix, X::AbstractMatrix)
    @assert size(X, 1) == size(X, 2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, P, X)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, P::PermutationMatrix{T}, X::AbstractMatrix) where {T}
    ind = getind(P)
    @inbounds @simd for k in 1:size(P, 1)
        @inbounds @simd for ki in 1:size(P, 1)
            Y[k, ki] = X[ind[k], ki]
        end
    end
end

function Base.:*(P::Adjoint{T, PermutationMatrix{T}}, X::AbstractMatrix) where {T}
    @assert size(X, 1) == size(X, 2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, P, X)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, P::Adjoint{T, PermutationMatrix{T}}, X::AbstractMatrix) where {T}
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:size(P, 1)
        @inbounds @simd for ki in 1:size(P, 1)
            Y[ind[k], ki] = X[k, ki]
        end
    end
end

function Base.:*(P::Transpose{T, PermutationMatrix{T}}, X::AbstractMatrix) where {T}
    @assert size(X, 1) == size(X, 2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, P, X)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, P::Transpose{T, PermutationMatrix{T}}, X::AbstractMatrix) where {T}
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:size(P, 1)
        @inbounds @simd for ki in 1:size(P, 1)
            Y[ind[k], ki] = X[k, ki]
        end
    end
end

# transposedvector - matrix product
Base.:*(x::Adjoint{T, S}, P::PermutationMatrix) where {T, S <: AbstractVector} = (P' * x')'
Base.:*(x::Adjoint{T, S}, P::Adjoint{T1, PermutationMatrix{T1}}) where {T, T1, S <: AbstractVector} = (P' * x')'
Base.:*(x::Adjoint{T, S}, P::Transpose{T1, PermutationMatrix{T1}}) where {T, T1, S <: AbstractVector} = (P' * x')'
Base.:*(x::Transpose{T, S}, P::PermutationMatrix) where {T, S <: AbstractVector} = transpose(transpose(P) * transpose(x))
Base.:*(x::Transpose{T, S}, P::Adjoint{T1, PermutationMatrix{T1}}) where {T, T1, S <: AbstractVector} = transpose(transpose(P) * transpose(x))
Base.:*(x::Transpose{T, S}, P::Transpose{T1, PermutationMatrix{T1}}) where {T, T1, S <: AbstractVector} = transpose(transpose(P) * transpose(x))

# matrix-Permutation multiplication
function Base.:*(X::AbstractMatrix, P::PermutationMatrix)
    @assert size(X, 1) == size(X, 2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, X, P)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, P::PermutationMatrix)
    ind = getind(P)
    @inbounds @simd for k in 1:size(P, 1)
        @inbounds @simd for ki in 1:size(P, 1)
            Y[ki, ind[k]] = X[ki, k]
        end
    end
end

function Base.:*(X::AbstractMatrix, P::Adjoint{T, PermutationMatrix{T}}) where {T}
    @assert size(X, 1) == size(X, 2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, X, P)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, P::Adjoint{T, PermutationMatrix{T}}) where {T}
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:size(P, 1)
        @inbounds @simd for ki in 1:size(P, 1)
            Y[ki, k] = X[ki, ind[k]]
        end
    end
end

function Base.:*(X::AbstractMatrix, P::Transpose{T, PermutationMatrix{T}}) where {T}
    @assert size(X, 1) == size(X, 2) "Multiplication with permutation matrices is only supported for square matrices."
    Y = similar(X)
    mul!(Y, X, P)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, P::Transpose{T, PermutationMatrix{T}}) where {T}
    ind = getind(P.parent) # explicitly take the index of the parent as not to call sortperm
    @inbounds @simd for k in 1:size(P, 1)
        @inbounds @simd for ki in 1:size(P, 1)
            Y[ki, k] = X[ki, ind[k]]
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
    @inbounds @simd for k1 in 1:size(P, 1)
        @inbounds @simd for k2 in 1:size(P, 1)
            Y[ind[k1], ind[k2]] = X[k1, k2]
        end
    end
end
