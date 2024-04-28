
@marginalrule InverseWishart(:out_ν_S) (m_out::InverseWishartFast, m_ν::PointMass, m_S::PointMass) = begin
    q_out = prod(ClosedProd(), InverseWishartFast(mean(m_ν), mean(m_S)), m_out)
    return convert_paramfloattype((out = convert(InverseWishart, q_out), ν = m_ν, S = m_S))
end
