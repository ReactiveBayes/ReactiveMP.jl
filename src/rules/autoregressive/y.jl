
@rule AR(:y, Marginalisation) (m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::Any, meta::ARMeta) = begin
    mθ, Vθ = mean_cov(q_θ)
    mx, Wx = mean_invcov(m_x)

    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(meta), getorder(meta), mγ)

    D = Wx + mγ * Vθ
    C = mA * inv(D)

    my = C * Wx * mx
    Vy = add_transition!(C * mA', mV)
    
    return convert(promote_variate_type(getvform(meta), NormalMeanVariance), my, Vy)
end

@rule AR(:y, Marginalisation) (q_x::Any, q_θ::Any, q_γ::Any, meta::ARMeta) = begin
    mA = as_companion_matrix(mean(q_θ))

    mV = ar_transition(getvform(meta), getorder(meta), mean(q_γ))

    return convert(promote_variate_type(getvform(meta), NormalMeanVariance), mA*mean(q_x), mV)
end