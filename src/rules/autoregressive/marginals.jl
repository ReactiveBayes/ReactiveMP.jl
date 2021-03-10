export marginalrule

@marginalrule AR(:y_x) (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta) = begin
    return ar_y_x_marginal(getstype(meta), m_y, m_x, q_θ, q_γ, meta)
end

function ar_y_x_marginal(::ARsafe, m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta)
    mθ, Vθ = mean(q_θ), cov(q_θ)

    mA = as_companion_matrix(mθ)
    mW = ar_precision(getvform(meta), getorder(meta), mean(q_γ))

    b_my, b_Vy = mean(m_y), cov(m_y)
    f_mx, f_Vx = mean(m_x), cov(m_x)

    D = cholinv(f_Vx) + mean(q_γ) * Vθ
    W = [cholinv(b_Vy)+mW -mW*mA; -(mA'*mW) D+mA'*mW*mA]
    m = cholinv(W)*[cholinv(b_Vy)*b_my; cholinv(f_Vx)*f_mx]

    return MvNormalMeanPrecision(m, W)
end

function ar_y_x_marginal(::ARunsafe, m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta)
    mθ, Vθ = mean(q_θ), cov(q_θ)

    mA = as_companion_matrix(mθ)
    invmA = inv(mA)
    mγ = mean(q_γ)
    mV = ar_transition(getvform(meta), getorder(meta), mγ)

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