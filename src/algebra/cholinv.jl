export cholinv, cholsqrt

using PositiveFactorizations

cholinv(x)           = inv(cholesky(PositiveFactorizations.Positive, Hermitian(x)))
cholinv(x::Diagonal) = Diagonal(inv.(diag(x)))

cholsqrt(x)           = Matrix(cholesky(PositiveFactorizations.Positive, x).L)
cholsqrt(x::Diagonal) = Diagonal(sqrt.(diag(x)))