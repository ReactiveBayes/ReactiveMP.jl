
@rule Beta(:out, Marginalisation) (m_α::PointMass, m_β::PointMass) = Beta(mean(m_α), mean(m_β))

@rule Beta(:out, Marginalisation) (q_α::PointMass, q_β::PointMass) = Beta(mean(q_α), mean(q_β))
