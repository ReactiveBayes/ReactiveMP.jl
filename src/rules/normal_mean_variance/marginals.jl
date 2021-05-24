
@marginalrule NormalMeanVariance(:out_μ_v) (m_out::NormalDistributionsFamily, m_μ::PointMass, m_v::PointMass) = begin
    return (out = prod(ProdAnalytical(), NormalMeanVariance(mean(m_μ), mean(m_v)), m_out), μ = m_μ, v = m_v)
end

@marginalrule NormalMeanVariance(:out_μ_v) (m_out::PointMass, m_μ::NormalDistributionsFamily, m_v::PointMass) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), m_μ, NormalMeanVariance(mean(m_out), mean(m_v))), v = m_v)
end