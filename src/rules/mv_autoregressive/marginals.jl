
@marginalrule MAR(:y_x) (
    m_y::MultivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta
) = begin
    return ar_y_x_marginal(m_y, m_x, q_a, q_Λ, meta)
end

function ar_y_x_marginal(
    m_y::MultivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta
)
    order, ds = getorder(meta), getdimensionality(meta)
    F = Multivariate
    dim = order * ds

    ma, Va = mean_cov(q_a)
    mΛ = mean(q_Λ)

    mA = mar_companion_matrix(order, ds, ma)
    mW = mar_transition(getorder(meta), mΛ)

    b_my, b_Vy = mean_cov(m_y)
    f_mx, f_Vx = mean_cov(m_x)

    inv_b_Vy = cholinv(b_Vy)
    inv_f_Vx = cholinv(f_Vx)

    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:ds]
    Fs = [mask_mar(order, ds, i) for i in 1:ds]

    Ξ = inv_f_Vx + sum(sum(es[j]' * mW * es[i] * Fs[j] * Va * Fs[i]' for i in 1:ds) for j in 1:ds)

    W_11 = inv_b_Vy + mW

    # negate_inplace!(mW * mA)
    W_12 = -(mW * mA)

    W_21 = -(mA' * mW)

    W_22 = Ξ + mA' * mW * mA

    W = [W_11 W_12; W_21 W_22]
    ξ = [inv_b_Vy * b_my; inv_f_Vx * f_mx]

    return MvNormalWeightedMeanPrecision(ξ, W)
end
