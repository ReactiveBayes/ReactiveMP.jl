
@marginalrule Beta(:out_a_b) (m_out::Beta, m_a::PointMass, m_b::PointMass) = begin
    return convert_paramfloattype((out = prod(ProdAnalytical(), Beta(mean(m_a), mean(m_b)), m_out), a = m_a, b = m_b))
end
