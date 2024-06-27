# VMP: Stuctured
@rule ContinuousTransition(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma = mean(q_a)
    mx, Vx = mean_cov(m_x)

    mW = mean(q_W)

    epsilon = sqrt.(var(q_a))
    mA = ctcompanion_matrix(ma, epsilon, meta)

    Vy = mA * Vx * mA' + cholinv(mW)
    my = mA * mx

    return MvNormalMeanCovariance(my, Vy)
end

# VMP: Mean-field
@rule ContinuousTransition(:y, Marginalisation) (q_x::Any, q_a::Any, q_W::Any, meta::CTMeta) = MvNormalMeanPrecision(
    ctcompanion_matrix(mean(q_a), sqrt.(var(q_a)), meta) * mean(q_x), mean(q_W)
)
