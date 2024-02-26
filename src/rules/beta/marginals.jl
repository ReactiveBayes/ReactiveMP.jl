
@marginalrule Beta(:out_α_β) (m_out::Beta, m_α::PointMass, m_β::PointMass) = begin
    return convert_paramfloattype((out = prod(ClosedProd(), Beta(mean(m_α), mean(m_β)), m_out), a = m_α, b = m_β))
end
