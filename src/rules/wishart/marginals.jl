export marginalrule

@marginalrule Wishart(:out_ν_S) (m_out::WishartDistributionsFamily, m_ν::PointMass, m_S::PointMass) = begin
    q_out = prod(ProdAnalytical(), WishartMessage(mean(m_ν), mean(m_S)), m_out)
    return (out = convert(Wishart, q_out), ν = m_ν, S = m_S)
end
