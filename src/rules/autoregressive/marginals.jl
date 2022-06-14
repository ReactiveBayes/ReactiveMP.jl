
@marginalrule AR(:y_x) (
    m_y::NormalDistributionsFamily,
    m_x::NormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::Any,
    meta::ARMeta
) = begin
    return ar_y_x_marginal(getstype(meta), m_y, m_x, q_θ, q_γ, meta)
end

function ar_y_x_marginal(
    ::ARsafe,
    m_y::NormalDistributionsFamily,
    m_x::NormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::Any,
    meta::ARMeta
)
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mW = ar_precision(getvform(meta), getorder(meta), mγ)

    b_my, b_Vy = mean_cov(m_y)
    f_mx, f_Vx = mean_cov(m_x)

    inv_b_Vy = cholinv(b_Vy)
    inv_f_Vx = cholinv(f_Vx)

    D = inv_f_Vx + mγ * Vθ

    W_11 = add_precision(inv_b_Vy, mW)

    # Equvalent to -(mW * mA)
    W_12 = negate_inplace!(mW * mA)

    # Equivalent to (-mA' * mW)
    W_21 = negate_inplace!(mA' * mW)

    W_22 = D + mA' * mW * mA

    W = [W_11 W_12; W_21 W_22]
    ξ = [inv_b_Vy * b_my; inv_f_Vx * f_mx]

    return MvNormalWeightedMeanPrecision(ξ, W)
end

function ar_y_x_marginal(
    ::ARunsafe,
    m_y::NormalDistributionsFamily,
    m_x::NormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::Any,
    meta::ARMeta
)
    mθ, Vθ = mean(q_θ), cov(q_θ)

    mA = as_companion_matrix(mθ)
    invmA = inv(mA)
    mγ = mean(q_γ)
    mV = ar_transition(getvform(meta), getorder(meta), mγ)

    b_my, b_Vy = mean(m_y), cov(m_y)
    f_mx, f_Vx = mean(m_x), cov(m_x)

    E = mV - mV * inv(b_Vy + mV) * mV
    F = mV + mV * invmA' * (inv(f_Vx) + mγ * Vθ) * invmA' * mV
    ABDC = E - E * inv(F + E) * E
    BD = -invmA' + invmA' * inv(invmA * mV * invmA' + inv((inv(f_Vx) + mγ * Vθ))) * invmA * mV * invmA'
    DC = -invmA + invmA * mV * invmA' * inv(invmA * mV * invmA' + inv((inv(f_Vx) + mγ * Vθ))) * invmA
    D =
        invmA * mV * invmA' -
        invmA * mV * invmA' * inv(invmA * mV * invmA' + inv((inv(f_Vx) + mγ * Vθ))) * invmA * mV * invmA'
    invW = [ABDC -ABDC*BD; -DC*ABDC D+DC*ABDC*BD]

    m = invW * [inv(b_Vy) * b_my; inv(f_Vx) * f_mx]

    return MvNormalMeanCovariance(m, invW)
end
