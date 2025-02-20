@rule DirichletCollection(:out, Marginalisation) (m_a::PointMass,) = DirichletCollection(mean(m_a))

@rule DirichletCollection(:out, Marginalisation) (q_a::PointMass,) = DirichletCollection(mean(q_a))
