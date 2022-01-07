export cholinv, cholsqrt

using LinearAlgebra
using PositiveFactorizations

import LinearAlgebra: BlasInt

cholinv(x)           = inv(fastcholesky(x))
cholinv(x::Diagonal) = Diagonal(inv.(diag(x)))
cholinv(x::Real)     = inv(x)

function cholinv(x::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat }
    y = fastcholesky(x)
    LinearAlgebra.inv!(y)
    return y.factors
end

cholsqrt(x)           = Matrix(fastcholesky(x).L)
cholsqrt(x::Diagonal) = Diagonal(sqrt.(diag(x)))
cholsqrt(x::Real)     = sqrt(x)

chollogdet(x)           = logdet(fastcholesky(x))
chollogdet(x::Diagonal) = logdet(x)
chollogdet(x::Real)     = logdet(x)

function cholinv_logdet(x) 
    # calculate cholesky decomposition
    y = fastcholesky(x)
      
    # return inverse and log-determinant
    return inv(y), logdet(y)
end
function cholinv_logdet(x::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat } 
    # calculate cholesky decomposition
    y = fastcholesky(x)
    
    # calculate logdeterminant of cholesky decomposition
    ly = logdet(y)
    
    # calculate inplace inverse of A and store in y.factors
    LinearAlgebra.inv!(y)
    
    # return inverse and log-determinant
    return y.factors, ly
end
cholinv_logdet(x::Diagonal) = Diagonal(inv.(diag(x))), mapreduce(z -> log(z), +, diag(x))
cholinv_logdet(x::Real)     = inv(x), log(abs(x))

function fastcholesky(mat::AbstractMatrix)
    try
        A = copy(mat)
        return fastcholesky!(A)
    catch
        return cholesky(PositiveFactorizations.Positive, Hermitian(mat))
    end
    
end

function fastcholesky!(A::AbstractMatrix)
    nbrows, nbcols = size(A)
    @inbounds for col=1:nbcols
        @inbounds @simd for idx in 1:col-1
            A[col, col] -= A[col, idx]^2;
        end
        A[col, col] = sqrt(A[col, col])

        @inbounds for row in col+1: nbrows
            @inbounds @simd for idx in 1:col-1
                A[row, col] -= A[row, idx]*A[col, idx]
            end
            A[row, col] /= A[col, col]
        end
    end
    return Cholesky(A, 'L', convert(BlasInt, 0))
end

function fastcholesky!(A::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat }
    # blocked version (https://arxiv.org/pdf/1812.02056.pdf)

    # step size
    s = 250

    n = size(A,1)
    z = 1
    @inbounds for c in 1:n

        if c == z + s
            BLAS.gemm!('N', 'T', -one(T), view(A, c:n, z:c-1), view(A, c:n, z:c-1), one(T), view(A, c:n, c:n))
            z = c
        end

        @inbounds for k in z:c-1
            A[c,c] -= A[c,k]^2
        end
        A[c,c] = sqrt(A[c,c])
        @inbounds for i in c+1:n
            @inbounds for k in z:c-1
                A[i,c] -= A[i,k]*A[c,k]
            end
            A[i,c] /= A[c,c]
        end
    end

    return Cholesky(A, 'L', convert(BlasInt, 0))
end