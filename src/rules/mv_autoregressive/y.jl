@rule MAR(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin
    ma, Va = mean_cov(q_a)
    mx, Wx = mean_invcov(m_x)

    mΛ = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)
    Fs, es    = getmasks(meta), getunits(meta)

    mA = mar_companion_matrix(ma, meta)
    mW = mar_transition(getorder(meta), mΛ)

    Λ = sum(sum(es[j]' * mW * es[i] * Fs[j] * Va * Fs[i]' for i in 1:ds) for j in 1:ds)

    Ξ = Λ + Wx
    z = Wx * mx

    Vy = mA * inv(mW) * mA' + inv(mW)
    my = mA * inv(Ξ) * z

    return MvNormalMeanCovariance(my, Vy)

end

@rule MAR(:y, Marginalisation) (q_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin
    order, ds = getorder(meta), getdimensionality(meta)

    mA = mar_companion_matrix(mean(q_a), meta)
    mW = mar_transition(getorder(meta), mean(q_Λ))

    return MvNormalMeanPrecision(mA * mean(q_x), mW)
end
