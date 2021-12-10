export cholinv, cholsqrt

using LinearAlgebra

cholinv(x)           = inv(cholesky(Hermitian(x)))
cholinv(x::Diagonal) = Diagonal(inv.(diag(x)))
cholinv(x::Real)     = inv(x)
function cholinv(x::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat }
    y = cholesky(Hermitian(x))
    LinearAlgebra.inv!(y)
    return y.factors
end

cholsqrt(x)           = Matrix(cholesky(Hermitian(x)).L)
cholsqrt(x::Diagonal) = Diagonal(sqrt.(diag(x)))
cholsqrt(x::Real)     = sqrt(x)

function cholinv_logdet(x::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat } 
    # calculate cholesky decomposition
    y = cholesky(Hermitian(A))
    
    # calculate logdeterminant of cholesky decomposition
    L = y.L
    z = zero(eltype(y))
    @inbounds @simd for k ∈ 1:size(L,1)
        z += log(L[k,k])
    end
    z *= 2
    
    # return inverse and log-determinant
    return inv(y), z
end
function cholinv_logdet(x::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat } 
    # calculate cholesky decomposition
    y = cholesky(Hermitian(A))
    
    # calculate logdeterminant of cholesky decomposition
    L = y.L
    z = zero(eltype(y))
    @inbounds @simd for k ∈ 1:size(L,1)
        z += log(L[k,k])
    end
    z *= 2
    
    # calculate inplace inverse of A and store in y.factors
    LinearAlgebra.inv!(y)
    
    # return inverse and log-determinant
    return y.factors, z
end
cholinv_logdet(x::Diagonal) = Diagonal(inv.(diag(x))), mapreduce(z -> log(z), +, diag(x))
cholinv_logdet(x::Real)     = inv(x), log(abs(x))
