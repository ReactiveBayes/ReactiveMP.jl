export rule

@rule Beta(:out, Marginalisation) (m_a::PointMass, m_b::PointMass) = Beta(mean(m_a), mean(m_b))