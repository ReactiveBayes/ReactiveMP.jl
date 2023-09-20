
@marginalrule Dirichlet(:out_a) (m_out::Dirichlet, m_a::PointMass) = begin
    return convert_paramfloattype((out = prod(ProdAnalytical(), Dirichlet(mean(m_a)), m_out), a = m_a))
end
