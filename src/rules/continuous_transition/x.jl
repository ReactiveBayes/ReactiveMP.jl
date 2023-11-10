@rule ContinuousTransition(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(m_y)

    mW = mean(q_W)

    dy, dx = getdimensionality(meta)
    Fs, es = getmasks(meta, ma), getunits(meta)

    mA = ctcompanion_matrix(ma, meta)

    W = sum(sum(es[j]' * mW * es[i] * Fs[j] * Va * Fs[i]' for i in 1:length(Fs)) for j in 1:length(Fs))

    Σ₁ = Hermitian(pinv(mA) * (Vy) * pinv(mA') + pinv(mA' * mW * mA))

    Ξ = (pinv(Σ₁) + W)
    z = pinv(Σ₁) * pinv(mA) * my

    return MvNormalWeightedMeanPrecision(z, Ξ)
end
