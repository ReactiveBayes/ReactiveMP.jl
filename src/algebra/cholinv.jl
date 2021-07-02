export cholinv, cholsqrt

using LinearAlgebra
using PositiveFactorizations

cholinv(x)           = inv(cholesky(PositiveFactorizations.Positive, Hermitian(x)))
cholinv(x::Diagonal) = Diagonal(inv.(diag(x)))
cholinv(x::Real)     = inv(x)

cholsqrt(x)           = Matrix(cholesky(PositiveFactorizations.Positive, x).L)
cholsqrt(x::Diagonal) = Diagonal(sqrt.(diag(x)))
cholsqrt(x::Real)     = sqrt(x)


