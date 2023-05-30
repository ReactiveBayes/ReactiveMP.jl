
@marginalrule Poisson(:out_l) (m_out::PointMass, m_l::Gamma) = begin
    return convert_paramfloattype((out = m_out, l = prod(ProdAnalytical(), Gamma(mean(m_out) + 1, 1), m_l)))
end
