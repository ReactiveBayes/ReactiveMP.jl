
@marginalrule Transfominator(:y_x) (
    m_y::MultivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_h::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::TMeta
) = begin
    return transfominator_marginal(m_y, m_x, q_h, q_Λ, meta)
end

function transfominator_marginal(
    m_y::MultivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_h::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::TMeta
)
    dx, dy = getdimensionality(meta)
    Fs, es = getmasks(meta), getunits(meta)

    mh, Vh = mean_cov(q_h)
    mΛ = mean(q_Λ)

    mH = tcompanion_matrix(mh, meta)

    b_my, b_Vy = mean_cov(m_y)
    f_mx, f_Vx = mean_cov(m_x)

    inv_b_Vy = cholinv(b_Vy)
    inv_f_Vx = cholinv(f_Vx)

    Ξ = inv_f_Vx + sum(sum(es[j]' * mΛ * es[i] * Fs[j] * Vh * Fs[i]' for i in 1:length(Fs)) for j in 1:length(Fs))

    W_11 = inv_b_Vy + mΛ

    # negate_inplace!(mW * mH)
    W_12 = -(mΛ * mH)

    W_21 = -(mH' * mΛ)

    W_22 = Ξ + mH' * mΛ * mH

    W = [W_11 W_12; W_21 W_22]
    ξ = [inv_b_Vy * b_my; inv_f_Vx * f_mx]

    return MvNormalWeightedMeanPrecision(ξ, W)
end
