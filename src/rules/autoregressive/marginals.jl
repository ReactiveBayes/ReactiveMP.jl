export marginalrule

@marginalrule AR(:y_x) (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily,
                        q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta{F, S}) where {F, S <: ARsafe} = begin
    mθ, Vθ = mean(q_θ), cov(q_θ)

    mA = as_CMatrix(mθ)
    mW = wMatrix(mean(q_γ), meta.order, F)

    b_my, b_Vy = mean(m_y), cov(m_y)
    f_mx, f_Vx = mean(m_x), cov(m_x)

    D = cholinv(f_Vx) + mean(q_γ) * Vθ
    W = [cholinv(b_Vy)+mW -mW*mA; -(mA'*mW) D+mA'*mW*mA]
    m = cholinv(W)*[cholinv(b_Vy)*b_my; cholinv(f_Vx)*f_mx]

    return MvNormalMeanPrecision(m, W)
end
