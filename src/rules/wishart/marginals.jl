
@marginalrule Wishart(:out_ν_S) (m_out::WishartDistributionsFamily, m_ν::PointMass, m_S::PointMass) = begin
    q_out = prod(ClosedProd(), WishartFast(mean(m_ν), cholinv(mean(m_S))), m_out)
    return convert_paramfloattype((out = convert(Wishart, q_out), ν = m_ν, S = m_S))
end
