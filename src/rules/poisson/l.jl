export rule

@rule Poisson(:l, Marginalisation) (m_out::PointMass,) = Gamma(mean(m_out) + 1, 1)

@rule Poisson(:l, Marginalisation) (q_out::Any,) = Gamma(mean(q_out) + 1, 1)
