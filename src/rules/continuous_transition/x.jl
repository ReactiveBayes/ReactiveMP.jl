@rule ContinuousTransition(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma, Va = mean_cov(q_a)
    my, Wy = mean_precision(m_y)
    mW = mean(q_W)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    epsilon = sqrt.(var(q_a))
    mA = ctcompanion_matrix(ma, epsilon, meta)

    # Woodbury identity
    # inv(inv(Wy) + inv(mW)) = Wy - Wy * inv(Wy + mW) * Wy
    WymW = Wy - Wy * cholinv(Wy + mW) * Wy
    Ξ = mA' * WymW * mA

    for (i, j) in Iterators.product(1:dy, 1:dy)
        Ξ += mW[j, i] * Fs[j] * Va * Fs[i]'
    end

    z = mA' * WymW * my

    return MvNormalWeightedMeanPrecision(z, Ξ)
end
