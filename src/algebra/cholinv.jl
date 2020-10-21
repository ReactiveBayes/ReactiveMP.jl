export cholinv

cholinv(x)           = inv(cholesky(Hermitian(x)))
cholinv(x::Diagonal) = Diagonal(inv.(diag(x)))