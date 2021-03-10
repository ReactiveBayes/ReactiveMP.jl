export rule

@rule AR(:y, Marginalisation) (m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta) = begin
    mθ, Vθ = mean(q_θ), cov(q_θ)
    mx, Vx = mean(m_x), cov(m_x)
    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(meta), getorder(meta), mγ)

    D = inv(Vx) + mean(q_γ)*Vθ

    my = mA*inv(D)*inv(Vx)*mx
    Vy = mA*inv(D)*mA' + mV
    
    return convert(promote_variate_type(getvform(meta), NormalMeanVariance), my, Vy)
end
