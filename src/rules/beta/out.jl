export rule

@rule Beta(:out, Marginalisation) (m_a::Dirac, m_b::Dirac) = Beta(mean(m_a), mean(m_b))