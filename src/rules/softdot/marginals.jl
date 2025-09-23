
# The following marginal rule is adaptation of the marginal rule for Autoregressive node
@marginalrule SoftDot(:y_x) (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_θ::Any, q_γ::Any) = begin
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)

    b_my, b_Vy = mean_cov(m_y)
    f_mx, f_Vx = mean_cov(m_x)

    inv_b_Vy = cholinv(b_Vy)
    inv_f_Vx = cholinv(f_Vx)

    D = inv_f_Vx + mγ * Vθ

    W_11 = inv_b_Vy + mγ

    W_12 = -mγ * mθ'

    W_21 = -mθ * mγ

    W_22 = D + mθ * mγ * mθ'

    W = [W_11 W_12; W_21 W_22]
    ξ = [inv_b_Vy * b_my; inv_f_Vx * f_mx]

    return MvNormalWeightedMeanPrecision(ξ, W)
end
