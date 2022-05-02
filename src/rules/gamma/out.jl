export rule

@rule Gamma(:out, Marginalisation) (m_α::PointMass, m_θ::PointMass) = Gamma(mean(m_α), mean(m_θ))

@rule Gamma(:out, Marginalisation) (q_α::PointMass, q_θ::PointMass) = Gamma(mean(q_α), mean(q_θ))
