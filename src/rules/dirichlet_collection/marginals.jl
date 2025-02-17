@marginalrule DirichletCollection(:out_a) (m_out::DirichletCollection, m_a::PointMass) = begin
    return convert_paramfloattype((out = prod(ClosedProd(), DirichletCollection(mean(m_a)), m_out), a = m_a))
end
