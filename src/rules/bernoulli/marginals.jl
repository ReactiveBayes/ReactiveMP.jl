
@marginalrule Bernoulli(:out_p) (m_out::PointMass, m_p::Beta) = begin
    r = mean(m_out)
    p = prod(ClosedProd(), Beta(one(r) + r, 2one(r) - r), m_p)
    return convert_paramfloattype((out = m_out, p = p))
end

@marginalrule Bernoulli(:out_p) (m_out::Bernoulli, m_p::PointMass) = begin
    return convert_paramfloattype((out = prod(ClosedProd(), Bernoulli(mean(m_p)), m_out), p = m_p))
end
