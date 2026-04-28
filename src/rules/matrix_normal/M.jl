
import Distributions: MatrixNormal, MatrixTDist
import ExponentialFamily: InverseWishartDistributionsFamily

@rule MatrixNormal(:M, Marginalisation) (m_out::PointMass, m_U::PointMass, m_V::PointMass) = MatrixNormal(
    mean(m_out), mean(m_U), mean(m_V)
)

@rule MatrixNormal(:M, Marginalisation) (
    m_out::MatrixNormal, m_U::PointMass, m_V::PointMass
) = begin
    X = mean(m_out)
    U = mean(m_U)
    V = mean(m_V)
    return MvNormalMeanCovariance(vec(X), kron(V, U) + cov(m_out))
end

@rule MatrixNormal(:M, Marginalisation) (
    m_out::PointMass,
    m_U::InverseWishartDistributionsFamily,
    m_V::PointMass,
) = begin
    X = mean(m_out)
    ν_U, Ψ_U = params(m_U)
    V = mean(m_V)
    n = size(X, 1)
    return MatrixTDist(ν_U - n + 1, X, Ψ_U, V)
end

@rule MatrixNormal(:M, Marginalisation) (
    m_out::PointMass,
    m_U::PointMass,
    m_V::InverseWishartDistributionsFamily,
) = begin
    X = mean(m_out)
    U = mean(m_U)
    ν_V, Ψ_V = params(m_V)
    p = size(X, 2)
    return MatrixTDist(ν_V - p + 1, X, U, Ψ_V)
end
