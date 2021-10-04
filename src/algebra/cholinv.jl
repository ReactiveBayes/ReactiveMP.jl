export cholinv, cholsqrt

using LinearAlgebra
using PositiveFactorizations

cholinv(x)           = inv(cholesky(PositiveFactorizations.Positive, Hermitian(x)))
cholinv(x::Diagonal) = inv(x)
cholinv(x::Real)     = inv(x)

cholsqrt(x)           = Matrix(cholesky(PositiveFactorizations.Positive, x).L)
cholsqrt(x::Diagonal) = sqrt(x)
cholsqrt(x::Real)     = sqrt(x)


