export marginalrule

@marginalrule Wishart(:out_ν_S) (m_out::Wishart, m_ν::Dirac, m_S::Dirac) = begin
    return (out = prod(ProdPreserveParametrisation(), Wishart(mean(m_ν), mean(m_S)), m_out), ν = m_ν, S = m_S)
end