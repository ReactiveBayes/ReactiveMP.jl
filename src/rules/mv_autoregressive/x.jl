using LinearAlgebra

@rule MAR(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin
    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(m_y)

    mΛ = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)
    Fs, es    = getmasks(meta), getunits(meta)
    dim       = order * ds

    mA = mar_companion_matrix(ma, meta)
    mW = mar_transition(getorder(meta), mΛ)

    Λ = sum(sum(es[j]' * mW * es[i] * Fs[j] * Va * Fs[i]' for i in 1:ds) for j in 1:ds)

    Σ₁ = Hermitian(pinv(mA) * (Vy) * pinv(mA') + pinv(mA' * mW * mA))
    Ξ = (pinv(Σ₁) + Λ)
    z = pinv(Σ₁) * pinv(mA) * my

    return MvNormalWeightedMeanPrecision(z, Ξ)
end

@rule MAR(:x, Marginalisation) (q_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin
    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(q_y)

    mΛ = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)
    Fs, es    = getmasks(meta), getunits(meta)

    mA = mar_companion_matrix(ma, meta)
    mW = mar_transition(getorder(meta), mΛ)

    Λ = sum(sum(es[j]' * mW * es[i] * Fs[j] * Va * Fs[i]' for i in 1:ds) for j in 1:ds)
    Λ₀ = Hermitian(mA' * mW * mA)

    Ξ = Λ₀ + Λ
    z = Λ₀ * (mA' \ my)

    return MvNormalWeightedMeanPrecision(z, Ξ)
end
