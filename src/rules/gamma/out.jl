export rule

@rule Gamma(:out, Marginalisation) (m_α::Dirac, m_θ::Dirac) = Gamma(mean(m_α), mean(m_θ))