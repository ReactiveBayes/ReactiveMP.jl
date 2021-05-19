export marginalrule

@marginalrule Wishart(:out_ν_S) (m_out::Wishart, m_ν::PointMass, m_S::PointMass) = begin
    return (out = prod(ProdAnalytical(), Wishart(mean(m_ν), mean(m_S)), m_out), ν = m_ν, S = m_S)
end