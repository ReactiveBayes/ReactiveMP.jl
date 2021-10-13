
@marginalrule NormalMeanVariance(:out_μ_v) (m_out::NormalDistributionsFamily, m_μ::PointMass, m_v::PointMass) = begin
    return (out = prod(ProdAnalytical(), NormalMeanVariance(mean(m_μ), mean(m_v)), m_out), μ = m_μ, v = m_v)
end

@marginalrule NormalMeanVariance(:out_μ_v) (m_out::PointMass, m_μ::NormalDistributionsFamily, m_v::PointMass) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), m_μ, NormalMeanVariance(mean(m_out), mean(m_v))), v = m_v)
end

@marginalrule NormalMeanVariance(:out_μ_v) (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily, m_v::PointMass) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_μ, W_μ     = weightedmean_precision(m_μ)

    W_bar = inv(mean(m_v))

    W  = [ W_out + W_bar -W_bar; -W_bar W_μ + W_bar ]
    xi = [ xi_out; xi_μ ]

    return (out_μ = MvNormalWeightedMeanPrecision(xi, W), v = m_v)
end

@marginalrule NormalMeanVariance(:out_μ) (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily, q_v::Any) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_μ, W_μ     = weightedmean_precision(m_μ)

    W_bar = inv(mean(q_v))

    W  = [ W_out + W_bar -W_bar; -W_bar W_μ + W_bar ]
    xi = [ xi_out; xi_μ ]

    return MvNormalWeightedMeanPrecision(xi, W)
end

@marginalrule NormalMeanVariance(:out_μ) (m_out::PointMass, m_μ::UnivariateNormalDistributionsFamily, q_v::Any) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), NormalMeanVariance(mean(m_out), mean(q_v)), m_μ))
end

@marginalrule NormalMeanVariance(:out_μ) (m_out::UnivariateNormalDistributionsFamily, m_μ::PointMass, q_v::Any) = begin
    return (out = prod(ProdAnalytical(), NormalMeanVariance(mean(m_μ), mean(q_v)), m_out), μ = m_μ)
end