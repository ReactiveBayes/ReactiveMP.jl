
@rule Beta(:out, Marginalisation) (m_a::PointMass, m_b::PointMass) = Beta(mean(m_a), mean(m_b))

@rule Beta(:out, Marginalisation) (q_a::PointMass, q_b::PointMass) = Beta(mean(q_a), mean(q_b))
