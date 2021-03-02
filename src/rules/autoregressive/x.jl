export rule

@rule AR(:x, Marginalisation) (m_y::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta{F}) where F = begin
    mθ, Vθ = mean(q_θ), cov(q_θ)
    my, Vy = mean(m_y), cov(m_y)

    mγ = mean(q_γ)

    mA = as_CMatrix(mθ)
    mV = transition(mγ, meta.order, F)

    D = mA'*inv(Vy + mV)*mA + mγ*Vθ
    Vx = inv(D)
    mx = inv(D)*mA'*inv(Vy + mV)*my

    return convert(promote_variate_type(F, NormalMeanVariance), mx, Vx)
end
