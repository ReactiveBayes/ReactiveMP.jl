export rule

@rule GammaShapeRate(:out, Marginalisation) (m_α::PointMass, m_β::PointMass) = GammaShapeRate(mean(m_α), mean(m_β))

@rule GammaShapeRate(:out, Marginalisation) (q_α::Any, q_β::Any) = GammaShapeRate(mean(q_α), mean(q_β))
