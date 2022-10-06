@rule Gamma(:out, Marginalisation) (m_α::PointMass, m_θ::PointMass) = Gamma(mean(m_α), mean(m_θ))

@rule Gamma(:out, Marginalisation) (q_α::Any, q_θ::Any) = Gamma(mean(q_α), mean(q_θ))
