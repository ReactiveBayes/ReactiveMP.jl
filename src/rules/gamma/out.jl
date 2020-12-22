export rule

@rule Gamma(:out, Marginalisation) (m_α::Dirac, m_θ::Dirac) = Gamma(mean(m_α), mean(m_θ))

@rule Gamma(:out, Marginalisation) (q_α::Dirac, q_θ::Dirac) = Gamma(mean(q_α), mean(q_θ))