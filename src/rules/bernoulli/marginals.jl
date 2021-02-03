export marginalrule

@marginalrule Bernoulli(:out_p) (m_out::PointMass, m_p::Beta) = begin
    r = mean(m_out)
    p = prod(ProdPreserveParametrisation(), Beta(one(r) + r, 2one(r) - r), m_p)
    return (out = m_out, p = p)
end