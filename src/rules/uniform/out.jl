
@rule Uniform(:out, Marginalisation) (m_a::PointMass, m_b::PointMass) = Uniform(mean(m_a), mean(m_b))
@rule Uniform(:out, Marginalisation) (q_a::PointMass, m_b::PointMass) = Uniform(mean(q_a), mean(m_b))
@rule Uniform(:out, Marginalisation) (m_a::PointMass, q_b::PointMass) = Uniform(mean(m_a), mean(q_b))
@rule Uniform(:out, Marginalisation) (q_a::PointMass, q_b::PointMass) = Uniform(mean(q_a), mean(q_b))
