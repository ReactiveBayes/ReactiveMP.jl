export CompanionMatrix, CompanionMatrixTransposed

import Base: *
import LinearAlgebra: transpose, inv

"""
    CompanionMatrix

Represents a matrix of the following structure:

θ1 θ2 θ3 ... θn-1 θn
 1  0  0  ...   0  0 
 0  1  0  ...   0  0 
 .  .  .  ...   .  .
 .  .  .  ...   .  .
 0  0  0  ...   0  0
 0  0  0  ...   1  0
"""
struct CompanionMatrix{R <: Real, T <: AbstractVector{R}} <: AbstractMatrix{R}
    θ::T
end

Base.eltype(::CompanionMatrix{R}) where {R} = R
Base.size(cmatrix::CompanionMatrix)         = (length(cmatrix.θ), length(cmatrix.θ))
Base.length(cmatrix::CompanionMatrix)       = prod(size(cmatrix))

Base.getindex(cmatrix::CompanionMatrix, i::Int) = getindex(cmatrix, map(r -> r + 1, reverse(divrem(i - 1, first(size(cmatrix)))))...)

function Base.getindex(cmatrix::CompanionMatrix, i::Int, j::Int)
    if i === 1
        return cmatrix.θ[j]
    elseif i === j + 1
        return one(eltype(cmatrix))
    else
        return zero(eltype(cmatrix))
    end
end

struct CompanionMatrixTransposed{R <: Real, T <: AbstractVector{R}} <: AbstractMatrix{R}
    θ::T
end

Base.eltype(::CompanionMatrixTransposed{R}) where {R} = R
Base.size(cmatrix::CompanionMatrixTransposed)         = (length(cmatrix.θ), length(cmatrix.θ))
Base.length(cmatrix::CompanionMatrixTransposed)       = prod(size(cmatrix))

Base.getindex(cmatrix::CompanionMatrixTransposed, i::Int) = getindex(cmatrix, map(r -> r + 1, reverse(divrem(i - 1, first(size(cmatrix)))))...)

function Base.getindex(cmatrix::CompanionMatrixTransposed, i::Int, j::Int)
    if j === 1
        return cmatrix.θ[i]
    elseif j === i + 1
        return one(eltype(cmatrix))
    else
        return zero(eltype(cmatrix))
    end
end

as_companion_matrix(θ::T) where {R, T <: AbstractVector{R}} = CompanionMatrix{R, T}(θ)
as_companion_matrix(θ::T) where {T <: Real}                 = θ

LinearAlgebra.transpose(cmatrix::CompanionMatrix)           = CompanionMatrixTransposed(cmatrix.θ)
LinearAlgebra.transpose(cmatrix::CompanionMatrixTransposed) = CompanionMatrix(cmatrix.θ)

LinearAlgebra.adjoint(cmatrix::CompanionMatrix)           = CompanionMatrixTransposed(cmatrix.θ)
LinearAlgebra.adjoint(cmatrix::CompanionMatrixTransposed) = CompanionMatrix(cmatrix.θ)

LinearAlgebra.inv(t::Union{CompanionMatrix, CompanionMatrixTransposed}) = inv(as_matrix(t))

function as_matrix(cmatrix::CompanionMatrix)
    dim     = first(size(cmatrix))
    S       = zeros(dim, dim)
    S[1, :] = cmatrix.θ
    for i in 2:dim
        S[i, i - 1] = one(eltype(cmatrix))
    end
    S
end

function as_matrix(cmatrix::CompanionMatrixTransposed)
    dim     = first(size(cmatrix))
    S       = zeros(dim, dim)
    S[:, 1] = cmatrix.θ
    for i in 2:dim
        S[i - 1, i] = one(eltype(cmatrix))
    end
    S
end

function Base.:*(tm::CompanionMatrix, v::AbstractVector)
    r = similar(v)
    r[1] = tm.θ' * v
    for i in 1:(length(v) - 1)
        r[i + 1] = v[i]
    end
    return r
end

function Base.:*(tm::CompanionMatrix, m::AbstractMatrix)
    l = length(tm.θ)
    r = similar(m)

    for (index, col) in enumerate(eachcol(m))
        r[1, index] = tm.θ' * col
        for i in 1:(l - 1)
            r[i + 1, index] = col[i]
        end
    end

    return r
end

function Base.:*(m::AbstractMatrix, tm::CompanionMatrixTransposed)
    l = length(tm.θ)
    r = similar(m)

    for (index, row) in enumerate(eachrow(m))
        r[index, 1] = tm.θ' * row
        for i in 1:(l - 1)
            r[index, i + 1] = row[i]
        end
    end

    return r
end

function Base.:*(m::AbstractMatrix, tm::CompanionMatrix)
    l = length(tm.θ)
    r = similar(m)

    return (tm' * m')'
end

function Base.:*(tm::CompanionMatrixTransposed, m::AbstractMatrix)
    l = length(tm.θ)
    r = similar(m)

    for (index, col) in enumerate(eachcol(m))
        for j in 1:(l - 1)
            r[j, index] = tm.θ[j] * col[1] + col[j + 1]
        end
        r[l, index] = tm.θ[l] * col[1]
    end

    return r
end
