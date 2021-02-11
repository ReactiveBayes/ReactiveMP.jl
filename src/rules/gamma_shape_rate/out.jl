export rule

@rule GammaShapeRate(:out, Marginalisation) (m_α::PointMass, m_β::PointMass) = Gamma(mean(m_α), mean(m_β))

@rule GammaShapeRate(:out, Marginalisation) (q_α::PointMass, q_β::PointMass) = Gamma(mean(q_α), mean(q_β))
