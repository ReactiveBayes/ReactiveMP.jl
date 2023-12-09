@rule ContinuousTransition(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma = mean(q_a)
    mx, Vx = mean_cov(m_x)

    mW = mean(q_W)

    dy, dx = getdimensionality(meta)
    Fs, es = getjacobians(meta, ma), getunits(meta)

    mA = ctcompanion_matrix(ma, sqrt.(var(q_a)), meta)

    Vy = mA * Vx * mA' + inv(mW)
    my = mA * mx

    return MvNormalMeanCovariance(my, Vy)
end
