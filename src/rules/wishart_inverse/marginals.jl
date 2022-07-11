export marginalrule

@marginalrule InverseWishart(:out_ν_S) (m_out::InverseWishartMessage, m_ν::PointMass, m_S::PointMass) = begin
    q_out = prod(ProdAnalytical(), InverseWishartMessage(mean(m_ν), mean(m_S)), m_out)
    return (out = convert(Wishart, q_out), ν = m_ν, S = m_S)
end
