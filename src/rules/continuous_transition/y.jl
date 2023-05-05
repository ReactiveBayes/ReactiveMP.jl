@rule ContinuousTransition(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, q_h::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::CTMeta) = begin
    mh, Vh = mean_cov(q_h)
    mx, Wx = mean_invcov(m_x)

    mΛ = mean(q_Λ)

    dy, dx = getdimensionality(meta)
    Fs, es = getmasks(meta), getunits(meta)

    mH = ctcompanion_matrix(mh, meta)

    Λ = sum(sum(es[j]' * mΛ * es[i] * Fs[j] * Vh * Fs[i]' for i in 1:length(Fs)) for j in 1:length(Fs))

    Ξ = Λ + Wx
    z = Wx * mx

    Vy = mH * inv(Ξ) * mH' + inv(mΛ)
    my = mH * inv(Ξ) * z

    return MvNormalMeanCovariance(my, Vy)
end
