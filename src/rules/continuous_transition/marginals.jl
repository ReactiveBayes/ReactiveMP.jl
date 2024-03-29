
@marginalrule ContinuousTransition(:y_x) (m_y::MultivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta) = begin
    return continuous_tranition_marginal(m_y, m_x, q_a, q_W, meta)
end

function continuous_tranition_marginal(m_y::MultivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta)
    ma, Va = mean_cov(q_a)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    mW = mean(q_W)

    epsilon = sqrt.(var(q_a))
    mA = ctcompanion_matrix(ma, epsilon, meta)

    xiy, Wy = weightedmean_precision(m_y)
    xix, Wx = weightedmean_precision(m_x)

    W_11 = Wy + mW

    # 
    W_12 = negate_inplace!(mW * mA)

    W_21 = negate_inplace!(mA' * mW)

    Ξ = Wx
    for (i, j) in Iterators.product(1:dy, 1:dy)
        Ξ += mW[j, i] * Fs[j] * Va * Fs[i]'
    end

    W_22 = Ξ + mA' * mW * mA

    W = [W_11 W_12; W_21 W_22]
    ξ = [xiy; xix]

    return MvNormalWeightedMeanPrecision(ξ, W)
end
