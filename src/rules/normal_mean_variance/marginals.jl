export marginalrule

@marginalrule NormalMeanVariance(:out_μ_v) (m_out::NormalMeanVariance, m_μ::Dirac, m_v::Dirac) = begin
    return (out = prod(ProdPreserveParametrisation(), NormalMeanVariance(mean(m_μ), mean(m_v)), m_out), μ = m_μ, v = m_v)
end

@marginalrule NormalMeanVariance(:out_μ_v) (m_out::Dirac, m_μ::NormalMeanVariance, m_v::Dirac) = begin
    return (out = m_out, μ = prod(ProdPreserveParametrisation(), m_μ, NormalMeanVariance(mean(m_out), mean(m_v))), v = m_v)
end