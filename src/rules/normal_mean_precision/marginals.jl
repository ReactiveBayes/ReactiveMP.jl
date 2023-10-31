
@marginalrule NormalMeanPrecision(:out_μ_τ) (m_out::UnivariateNormalDistributionsFamily, m_μ::PointMass, m_τ::PointMass) = begin
    return (out = prod(ClosedProd(), NormalMeanPrecision(mean(m_μ), mean(m_τ)), m_out), μ = m_μ, τ = m_τ)
end

@marginalrule NormalMeanPrecision(:out_μ_τ) (m_out::PointMass, m_μ::UnivariateNormalDistributionsFamily, m_τ::PointMass) = begin
    return (out = m_out, μ = prod(ClosedProd(), m_μ, NormalMeanPrecision(mean(m_out), mean(m_τ))), τ = m_τ)
end

@marginalrule NormalMeanPrecision(:out_μ_τ) (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily, m_τ::PointMass) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_μ, W_μ     = weightedmean_precision(m_μ)

    W_bar = mean(m_τ)

    W  = [W_out+W_bar -W_bar; -W_bar W_μ+W_bar]
    xi = [xi_out; xi_μ]

    return (out_μ = MvNormalWeightedMeanPrecision(xi, W), τ = m_τ)
end

@marginalrule NormalMeanPrecision(:out_μ) (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily, q_τ::Any) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_μ, W_μ     = weightedmean_precision(m_μ)

    W_bar = mean(q_τ)

    W  = [W_out+W_bar -W_bar; -W_bar W_μ+W_bar]
    xi = [xi_out; xi_μ]

    return MvNormalWeightedMeanPrecision(xi, W)
end

@marginalrule NormalMeanPrecision(:out_μ) (m_out::PointMass, m_μ::UnivariateNormalDistributionsFamily, q_τ::Any) = begin
    return (out = m_out, μ = prod(ClosedProd(), NormalMeanPrecision(mean(m_out), mean(q_τ)), m_μ))
end

@marginalrule NormalMeanPrecision(:out_μ) (m_out::UnivariateNormalDistributionsFamily, m_μ::PointMass, q_τ::Any) = begin
    return (out = prod(ClosedProd(), NormalMeanPrecision(mean(m_μ), mean(q_τ)), m_out), μ = m_μ)
end
