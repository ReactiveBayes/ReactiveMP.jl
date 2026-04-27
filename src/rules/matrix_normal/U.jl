
import ExponentialFamily: InverseWishartFast

@rule MatrixNormal(:U, Marginalisation) (
    m_out::PointMass, m_M::PointMass, m_V::PointMass
) = begin
    X = mean(m_out)
    M = mean(m_M)
    V = mean(m_V)
    n, p = size(X)
    D = X - M
    Ψ = D * cholinv(V) * D'
    return InverseWishartFast(p - n - 1, Ψ)
end
