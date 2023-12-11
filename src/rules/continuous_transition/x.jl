@rule ContinuousTransition(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma, Va = mean_cov(q_a)
    my, Wy = mean_precision(m_y)
    mW = mean(q_W)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    epsilon = sqrt.(var(q_a))
    mA = ctcompanion_matrix(ma, epsilon, meta)

    W = sum(sum(StandardBasisVector(dy, j)' * mW * StandardBasisVector(dy, i) * Fs[j] * Va * Fs[i]' for i in 1:dy) for j in 1:dy)
    # Woodbury identity
    # inv(inv(Wy) + inv(mW)) = Wy - Wy * inv(Wy + mW) * Wy
    WymW = Wy - Wy * cholinv(Wy + mW) * Wy
    z = mA' * WymW * my
    Ξ = mA' * WymW * mA + W

    return MvNormalWeightedMeanPrecision(z, Ξ)
end
