export rule

@rule Wishart(:out, Marginalisation) (m_ν::PointMass, m_S::PointMass) = Wishart(mean(m_ν), mean(m_S))

@rule Wishart(:out, Marginalisation) (q_ν::Any, q_S::Any) = Wishart(mean(q_ν), mean(q_S))