
import Distributions: MatrixNormal, MatrixTDist
import ExponentialFamily: InverseWishartDistributionsFamily

@rule MatrixNormal(:out, Marginalisation) (
    m_M::PointMass, m_U::PointMass, m_V::PointMass
) = begin
    return MatrixNormal(mean(m_M), mean(m_U), mean(m_V))
end

@rule MatrixNormal(:out, Marginalisation) (
    m_M::MatrixNormal, m_U::PointMass, m_V::PointMass
) = begin
    M = mean(m_M)
    U = mean(m_U)
    V = mean(m_V)
    return MvNormalMeanCovariance(vec(M), kron(V, U) + cov(m_M))
end

@rule MatrixNormal(:out, Marginalisation) (
    m_M::PointMass, m_U::InverseWishartDistributionsFamily, m_V::PointMass
) = begin
    M = mean(m_M)
    ν_U, Ψ_U = params(m_U)
    V = mean(m_V)
    n = size(M, 1)
    return MatrixTDist(ν_U - n + 1, M, Ψ_U, V)
end

@rule MatrixNormal(:out, Marginalisation) (
    m_M::PointMass, m_U::PointMass, m_V::InverseWishartDistributionsFamily
) = begin
    M = mean(m_M)
    U = mean(m_U)
    ν_V, Ψ_V = params(m_V)
    p = size(M, 2)
    return MatrixTDist(ν_V - p + 1, M, U, Ψ_V)
end

@rule MatrixNormal(:out, Marginalisation) (
    q_M::MatrixNormal,
    q_U::Union{InverseWishartDistributionsFamily, PointMass},
    q_V::Union{InverseWishartDistributionsFamily, PointMass},
) = begin
    M = mean(q_M)
    U = cholinv(mean(cholinv, q_U))
    V = cholinv(mean(cholinv, q_V))
    return MatrixNormal(M, (U + U') / 2, (V + V') / 2)
end
