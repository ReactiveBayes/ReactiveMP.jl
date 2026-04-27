
import ExponentialFamily: InverseWishartFast

@rule MatrixNormal(:V, Marginalisation) (
    m_out::PointMass, m_M::PointMass, m_U::PointMass
) = begin
    X = mean(m_out)
    M = mean(m_M)
    U = mean(m_U)
    n, p = size(X)
    D = X - M
    Ψ = D' * cholinv(U) * D
    return InverseWishartFast(n - p - 1, Ψ)
end
