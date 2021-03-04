export CMatrix, CMatrixTran

import Base: *
import LinearAlgebra: transpose, inv

struct CMatrix{ R <: Real, T <: AbstractArray{R} }
    t :: T
end

struct CMatrixTran{ R <: Real, T <: AbstractArray{R} }
    t :: T
end

function as_CMatrix(t::T) where { R, T<:AbstractArray{R} }
    return CMatrix{R, T}(t)
end

function as_CMatrix(t::T) where { T<:Real }
    return t
end

LinearAlgebra.transpose(t::CMatrix)     = CMatrixTran(t.t)
LinearAlgebra.transpose(t::CMatrixTran) = CMatrix(t.t)

LinearAlgebra.adjoint(t::CMatrix)     = CMatrixTran(t.t)
LinearAlgebra.adjoint(t::CMatrixTran) = CMatrix(t.t)

function as_Matrix(t::CMatrix)
    dim = length(t.t)
    S = Matrix{Float64}(I, dim, dim)
    for i in dim:-1:2
           S[i,:] = S[i-1, :]
    end
    S[1, :] = zeros(dim)
    u = zeros(dim); u[1] = 1.0
    S + u*transpose(t.t)
end

function as_Matrix(t::CMatrixTran)
    return transpose(as_Matrix(transpose(t.t)))
end

function LinearAlgebra.inv(t::Union{CMatrix, CMatrixTran})
    inv(as_Matrix(t))
end

function Base.:*(tm::CMatrix, v::AbstractVector)
    r = similar(v)
    r[1] = tm.t' * v
    for i in 1:(length(v) - 1)
        r[i + 1] = v[i]
    end
    return r
end

function Base.:*(tm::CMatrix, m::AbstractMatrix)

    l = length(tm.t)
    r = similar(m)

    for (index, col) in enumerate(eachcol(m))
        r[1, index] = tm.t' * col
        for i in 1:(l - 1)
            r[i + 1, index] = col[i]
        end
    end

    return r
end

function Base.:*(m::AbstractMatrix, tm::CMatrixTran)

    l = length(tm.t)
    r = similar(m)

    for (index, row) in enumerate(eachrow(m))
        r[index, 1] = tm.t' * row
        for i in 1:(l - 1)
            r[index, i + 1] = row[i]
        end
    end

    return r
end

function Base.:*(m::AbstractMatrix, tm::CMatrix)

    l = length(tm.t)
    r = similar(m)

    return (tm' * m')'
end

function Base.:*(tm::CMatrixTran, m::AbstractMatrix)

    l = length(tm.t)
    r = similar(m)

    for (index, col) in enumerate(eachcol(m))
        for j in 1:l-1
            r[j, index] = tm.t[j] * col[1] + col[j + 1]
        end
        r[l, index] = tm.t[l] * col[1]
    end

    return r
end
