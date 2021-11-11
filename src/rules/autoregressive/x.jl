
@rule AR(:x, Marginalisation) (m_y::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta) = begin
    mθ, Vθ = mean_cov(q_θ)
    my, Vy = mean_cov(m_y)

    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(meta), getorder(meta), mγ)

    C = mA' * inv(add_transition(Vy, mV))

    W = C * mA + mγ * Vθ
    ξ = C * my

    return convert(promote_variate_type(getvform(meta), NormalWeightedMeanPrecision), ξ, W)
end
