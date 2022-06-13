export marginalrule

@marginalrule Poisson(:out_l) (m_out::PointMass, m_l::Gamma) = begin
    return (out = m_out, l = prod(ProdAnalytical(), Gamma(mean(m_out) + 1, 1), m_l))
end