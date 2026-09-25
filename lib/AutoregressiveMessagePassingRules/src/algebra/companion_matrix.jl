# The companion matrix of the AR coefficients θ, which shifts the lagged state down by one and
# puts θᵀx on top:
#
#     θ₁ θ₂ … θₙ₋₁ θₙ
#     1  0  …  0   0
#     0  1  …  0   0
#     ⋮        ⋱   ⋮
#     0  0  …  1   0
#
# and its transpose, as their own type so that `mA'` stays structured. Both are internal. Their
# products are written out only against dense `Matrix` and `Vector`, the combinations the rules
# form; any other operand takes the generic `AbstractMatrix` fallback, which reads `getindex`.
# Written only against concrete operands, no method of theirs is ambiguous.
struct CompanionMatrix{R <: Real, T <: AbstractVector{R}} <: AbstractMatrix{R}
    θ::T
end

struct CompanionMatrixTransposed{R <: Real, T <: AbstractVector{R}} <: AbstractMatrix{R}
    θ::T
end

const AnyCompanionMatrix = Union{CompanionMatrix, CompanionMatrixTransposed}

Base.size(A::AnyCompanionMatrix) = (length(A.θ), length(A.θ))

function Base.getindex(A::CompanionMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    i == 1 && return A.θ[j]
    return i == j + 1 ? one(eltype(A)) : zero(eltype(A))
end

function Base.getindex(A::CompanionMatrixTransposed, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    j == 1 && return A.θ[i]
    return j == i + 1 ? one(eltype(A)) : zero(eltype(A))
end

# The companion matrix of `θ`; for a univariate AR(1), `θ` itself.
as_companion_matrix(θ::AbstractVector{<:Real}) = CompanionMatrix(θ)
as_companion_matrix(θ::Real) = θ

LinearAlgebra.transpose(A::CompanionMatrix) = CompanionMatrixTransposed(A.θ)
LinearAlgebra.transpose(A::CompanionMatrixTransposed) = CompanionMatrix(A.θ)
LinearAlgebra.adjoint(A::CompanionMatrix) = CompanionMatrixTransposed(A.θ)
LinearAlgebra.adjoint(A::CompanionMatrixTransposed) = CompanionMatrix(A.θ)

LinearAlgebra.inv(A::CompanionMatrix) = inv(Matrix(A))
LinearAlgebra.inv(A::CompanionMatrixTransposed) = inv(Matrix(A))

function check_product_size(A, B)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("cannot multiply a matrix of size $(size(A)) by one of size $(size(B))"))
    return nothing
end

# A x: θᵀx on top, then x shifted down by one.
function Base.:*(A::CompanionMatrix, x::Vector)
    check_product_size(A, x)
    n = length(x)
    r = Vector{promote_type(eltype(A), eltype(x))}(undef, n)
    @inbounds r[1] = dot(A.θ, x)
    @inbounds for i in 1:(n - 1)
        r[i + 1] = x[i]
    end
    return r
end

# A M, column by column as above.
function Base.:*(A::CompanionMatrix, M::Matrix)
    check_product_size(A, M)
    n = length(A.θ)
    r = Matrix{promote_type(eltype(A), eltype(M))}(undef, n, size(M, 2))
    @inbounds for j in axes(M, 2)
        r[1, j] = dot(A.θ, view(M, :, j))
        for i in 1:(n - 1)
            r[i + 1, j] = M[i, j]
        end
    end
    return r
end

# M A: (M A)[i, j] = M[i, 1] θⱼ + M[i, j + 1], the last column without the second term.
function Base.:*(M::Matrix, A::CompanionMatrix)
    check_product_size(M, A)
    n = length(A.θ)
    r = Matrix{promote_type(eltype(M), eltype(A))}(undef, size(M, 1), n)
    @inbounds for j in 1:n, i in axes(M, 1)
        r[i, j] = M[i, 1] * A.θ[j] + (j < n ? M[i, j + 1] : zero(eltype(M)))
    end
    return r
end

# Aᵀ M: (Aᵀ M)[i, j] = θᵢ M[1, j] + M[i + 1, j], the last row without the second term.
function Base.:*(A::CompanionMatrixTransposed, M::Matrix)
    check_product_size(A, M)
    n = length(A.θ)
    r = Matrix{promote_type(eltype(A), eltype(M))}(undef, n, size(M, 2))
    @inbounds for j in axes(M, 2), i in 1:n
        r[i, j] = A.θ[i] * M[1, j] + (i < n ? M[i + 1, j] : zero(eltype(M)))
    end
    return r
end

# M Aᵀ: θᵀ against each row on the left, then the rows' entries shifted right by one.
function Base.:*(M::Matrix, A::CompanionMatrixTransposed)
    check_product_size(M, A)
    n = length(A.θ)
    r = Matrix{promote_type(eltype(M), eltype(A))}(undef, size(M, 1), n)
    @inbounds for i in axes(M, 1)
        r[i, 1] = dot(A.θ, view(M, i, :))
        for j in 2:n
            r[i, j] = M[i, j - 1]
        end
    end
    return r
end
