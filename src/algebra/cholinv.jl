export cholinv, cholsqrt

using LinearAlgebra
using PositiveFactorizations

cholinv(x)           = inv(cholesky(PositiveFactorizations.Positive, Hermitian(x)))
cholinv(x::Diagonal) = Diagonal(inv.(diag(x)))
cholinv(x::Real)     = inv(x)
function cholinv(x::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat }
    y = cholesky(PositiveFactorizations.Positive, Hermitian(x))
    LinearAlgebra.inv!(y)
    return y.factors
end

cholsqrt(x)           = Matrix(cholesky(PositiveFactorizations.Positive, x).L)
cholsqrt(x::Diagonal) = Diagonal(sqrt.(diag(x)))
cholsqrt(x::Real)     = sqrt(x)


