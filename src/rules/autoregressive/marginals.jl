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


 # NOTE: not safe for AR order > 1
@marginalrule AR(:y_x) (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily,
                        q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta{R, S}) where {R, S <: ARunsafe} = begin
    mθ, Vθ = mean(q_θ), cov(q_θ)

    mA = as_CMatrix(mθ)
    invmA = inv(mA)
    mγ = mean(q_γ)
    mV = transition(mγ, meta.order, R)

    b_my, b_Vy = mean(m_y), cov(m_y)
    f_mx, f_Vx = mean(m_x), cov(m_x)

    E = mV - mV*inv(b_Vy + mV)*mV
    F = mV + mV*invmA'*(inv(f_Vx) + mγ*Vθ)*invmA'*mV
    ABDC = E - E*inv(F + E)*E
    BD = -invmA' + invmA'*inv(invmA*mV*invmA' + inv((inv(f_Vx) + mγ*Vθ)))*invmA*mV*invmA'
    DC =  -invmA + invmA*mV*invmA'*inv(invmA*mV*invmA' + inv((inv(f_Vx) + mγ*Vθ)))*invmA
    D = invmA*mV*invmA' - invmA*mV*invmA'*inv(invmA*mV*invmA' + inv((inv(f_Vx) + mγ*Vθ)))*invmA*mV*invmA'
    invW = [ABDC -ABDC*BD; -DC*ABDC D+DC*ABDC*BD]

    m = invW*[inv(b_Vy)*b_my; inv(f_Vx)*f_mx]


    return MvNormalMeanCovariance(m, invW)
end
