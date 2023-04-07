@rule Transfominator(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily, q_h::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::TMeta) = begin
    mh, Vh = mean_cov(q_h)
    my, Vy = mean_cov(m_y)

    mΛ = mean(q_Λ)

    dy, dx = getdimensionality(meta)
    Fs, es = getmasks(meta), getunits(meta)

    mH = tcompanion_matrix(mh, meta)

    Λ = sum(sum(es[j]' * mΛ * es[i] * Fs[j] * Vh * Fs[i]' for i in 1:length(Fs)) for j in 1:length(Fs))

    Σ₁ = Hermitian(pinv(mH) * (Vy) * pinv(mH') + pinv(mH' * mΛ * mH))

    Ξ = (pinv(Σ₁) + Λ)
    z = pinv(Σ₁) * pinv(mH) * my

    return MvNormalWeightedMeanPrecision(z, Ξ)
end
