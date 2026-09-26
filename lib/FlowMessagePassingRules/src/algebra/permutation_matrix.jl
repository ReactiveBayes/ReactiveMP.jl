@doc raw"""
    PermutationMatrix(ind::Vector{<:Integer})
    PermutationMatrix([rng,] dim::Integer; switch_first::Bool = true)

A permutation matrix, with ones at `(k, ind[k])` for `k = 1:length(ind)` and zeros elsewhere, so
that `P * v` is `v[ind]`. Its inverse is its adjoint, and products with vectors, square matrices
and other permutations reorder entries without multiplying.

# Arguments

- `ind`: the permutation, a vector of the indices `1:n` in some order.
- `rng`: the generator the random permutation is drawn from. Default: the task's generator.
- `dim`: the size of the random permutation, `dim × dim`.

# Keywords

- `switch_first`: when `true`, a draw that keeps the first index in place swaps the first two
  entries, so the first element always moves. Default `true`.

# Throws

A `DimensionMismatch` for a product with a matrix that is not square.

# Examples

```jldoctest; setup = :(using FlowMessagePassingRules)
julia> PermutationMatrix([2, 1, 3]) * [10, 20, 30]
3-element Vector{Int64}:
 20
 10
 30
```
"""
struct PermutationMatrix{T <: Integer} <: AbstractMatrix{T}
    ind::Vector{T}
    function PermutationMatrix(ind::Vector{T}) where {T <: Integer}
        return new{T}(ind)
    end
end

function PermutationMatrix(rng::AbstractRNG, dim::T; switch_first::Bool = true) where {T <: Integer}
    ind = shuffle(rng, collect(1:dim))
    if switch_first && ind[1] == 1
        ind[1], ind[2] = ind[2], ind[1]
    end
    return PermutationMatrix(ind)
end

PermutationMatrix(dim::Integer; switch_first::Bool = true) = PermutationMatrix(default_rng(), dim; switch_first)

Base.size(mat::PermutationMatrix) = (length(mat.ind), length(mat.ind))
Base.size(mat::PermutationMatrix, d) = d::Integer <= 2 ? length(mat.ind) : 1
Base.length(mat::PermutationMatrix) = prod(size(mat))

Base.getindex(mat::PermutationMatrix, i::Int, j::Int) = mat.ind[i] == j ? one(eltype(mat)) : zero(eltype(mat))

LinearAlgebra.inv(mat::PermutationMatrix) = adjoint(mat)

# The permutation, its adjoint and its transpose, which for a permutation is its inverse.
const InversePermutation = Union{Adjoint{<:Integer, <:PermutationMatrix}, Transpose{<:Integer, <:PermutationMatrix}}
const AnyPermutation = Union{PermutationMatrix, InversePermutation}

# The matrices a permutation multiplies: dense ones and their adjoints and transposes. Not any
# `AbstractMatrix`, which a permutation is too, so that `P * X` and `X * P` do not both match a
# product of two permutations, which would be ambiguous.
const DenseOperand = Union{StridedMatrix, Adjoint{<:Any, <:StridedMatrix}, Transpose{<:Any, <:StridedMatrix}}

# Likewise for vectors: dense ones, since FillArrays' zeros, which BayesBase loads, have their own
# products with any matrix.
const DenseVector = StridedVector

# The indices of the ones: row `k` has its one in column `getind(P)[k]`.
getind(mat::PermutationMatrix) = mat.ind
getind(mat::InversePermutation) = sortperm(mat.parent.ind)

# A permutation times a vector.
Base.:*(P::AnyPermutation, v::DenseVector) = (y = similar(v); mul!(y, P, v); y)

function LinearAlgebra.mul!(y::AbstractVector, P::PermutationMatrix, v::DenseVector)
    ind = getind(P)
    @inbounds @simd for k in 1:size(P, 1)
        y[k] = v[ind[k]]
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, P::InversePermutation, v::DenseVector)
    ind = getind(P.parent) # the parent's indices, as not to call sortperm
    @inbounds @simd for k in 1:size(P, 1)
        y[ind[k]] = v[k]
    end
    return y
end

# A permutation times a square matrix, and a square matrix times a permutation.
function square_like(X::DenseOperand)
    size(X, 1) == size(X, 2) || throw(DimensionMismatch("multiplication with a permutation matrix is only supported for square matrices, got $(size(X))"))
    return similar(X)
end

Base.:*(P::AnyPermutation, X::DenseOperand) = (Y = square_like(X); mul!(Y, P, X); Y)
Base.:*(X::DenseOperand, P::AnyPermutation) = (Y = square_like(X); mul!(Y, X, P); Y)

function LinearAlgebra.mul!(Y::AbstractMatrix, P::PermutationMatrix, X::DenseOperand)
    ind = getind(P)
    @inbounds for ki in 1:size(P, 1), k in 1:size(P, 1)
        Y[k, ki] = X[ind[k], ki]
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, P::InversePermutation, X::DenseOperand)
    ind = getind(P.parent)
    @inbounds for ki in 1:size(P, 1), k in 1:size(P, 1)
        Y[ind[k], ki] = X[k, ki]
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::DenseOperand, P::PermutationMatrix)
    ind = getind(P)
    @inbounds for k in 1:size(P, 1), ki in 1:size(P, 1)
        Y[ki, ind[k]] = X[ki, k]
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::DenseOperand, P::InversePermutation)
    ind = getind(P.parent)
    @inbounds for k in 1:size(P, 1), ki in 1:size(P, 1)
        Y[ki, k] = X[ki, ind[k]]
    end
    return Y
end

# Two permutations: another permutation, with the indices composed.
Base.:*(a::AnyPermutation, b::AnyPermutation) = PermutationMatrix(getind(b)[getind(a)])
LinearAlgebra.mul!(Y::AbstractMatrix, a::AnyPermutation, b::AnyPermutation) = copyto!(Y, a * b)

# A row vector times a permutation.
Base.:*(x::Adjoint{<:Any, <:DenseVector}, P::AnyPermutation) = (P' * x')'
Base.:*(x::Transpose{<:Any, <:DenseVector}, P::AnyPermutation) = transpose(transpose(P) * transpose(x))

# Pᵀ X P for a square matrix `X`.
PT_X_P(X::AbstractMatrix, P::PermutationMatrix) = PT_X_P!(copy(X), X, P)

function PT_X_P!(Y::AbstractMatrix, X::AbstractMatrix, P::PermutationMatrix)
    ind = getind(P)
    @inbounds for k2 in 1:size(P, 1), k1 in 1:size(P, 1)
        Y[ind[k1], ind[k2]] = X[k1, k2]
    end
    return Y
end
