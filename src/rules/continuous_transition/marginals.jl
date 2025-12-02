
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

    # Optimized: factor out inner summation to reduce complexity from O(dy²) to O(dy)
    # Step 1: For each i, compute H[i] = Σⱼ mW[j,i] * Fs[j]
    H = [sum(mW[j, i] * Fs[j] for j in 1:dy) for i in 1:dy]

    # Step 2: Compute Ξ
    Ξ = Wx
    for i in 1:dy
        Ξ += H[i] * Va * Fs[i]'
    end

    W_22 = Ξ + mA' * mW * mA

    W = [W_11 W_12; W_21 W_22]
    ξ = [xiy; xix]

    return MvNormalWeightedMeanPrecision(ξ, W)
end
