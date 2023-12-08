
@marginalrule ContinuousTransition(:y_x) (m_y::MultivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta) = begin
    return continuous_tranition_marginal(m_y, m_x, q_a, q_W, meta)
end

function continuous_tranition_marginal(m_y::MultivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta)
    ma, Va = mean_cov(q_a)

    Fs, es = getjacobians(meta, ma), getunits(meta)

    mW = mean(q_W)

    mA = ctcompanion_matrix(ma, sqrt.(var(q_a)), meta)

    b_my, b_Vy = mean_cov(m_y)
    f_mx, f_Vx = mean_cov(m_x)

    inv_b_Vy = cholinv(b_Vy)
    inv_f_Vx = cholinv(f_Vx)

    Ξ = inv_f_Vx + sum(sum(es[j]' * mW * es[i] * Fs[j] * Va * Fs[i]' for i in 1:length(Fs)) for j in 1:length(Fs))

    W_11 = inv_b_Vy + mW

    # negate_inplace!(mW * mH)
    W_12 = -(mW * mA)

    W_21 = -(mA' * mW)

    W_22 = Ξ + mA' * mW * mA

    W = [W_11 W_12; W_21 W_22]
    ξ = [inv_b_Vy * b_my; inv_f_Vx * f_mx]

    return MvNormalWeightedMeanPrecision(ξ, W)
end
