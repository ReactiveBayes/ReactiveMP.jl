@rule ContinuousTransition(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma, Va = mean_cov(q_a)
    mx, Wx = mean_invcov(m_x)

    mW = mean(q_W)

    dy, dx = getdimensionality(meta)
    Fs, es = getmasks(meta, ma), getunits(meta)

    mA = ctcompanion_matrix(ma, meta)

    W = sum(sum(es[j]' * mW * es[i] * Fs[j] * Va * Fs[i]' for i in 1:length(Fs)) for j in 1:length(Fs))

    Ξ = W + Wx
    z = Wx * mx

    Vy = mA * inv(Ξ) * mA' + inv(mW)
    my = mA * inv(Ξ) * z

    return MvNormalMeanCovariance(my, Vy)
end
