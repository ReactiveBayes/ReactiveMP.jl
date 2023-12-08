@rule ContinuousTransition(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(m_y)

    mW = mean(q_W)

    dy, dx = getdimensionality(meta)
    Fs, es = getjacobians(meta, ma), getunits(meta)

    mA = ctcompanion_matrix(ma, sqrt.(var(q_a)), meta)

    W = sum(sum(es[j]' * mW * es[i] * Fs[j] * Va * Fs[i]' for i in 1:length(Fs)) for j in 1:length(Fs))

    z = mA' * inv(Vy + inv(mW)) * my
    Ξ = mA' * inv(Vy + inv(mW)) * mA + W

    return MvNormalWeightedMeanPrecision(z, Ξ)
end
