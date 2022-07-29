export rule

@rule NOT(:out, Marginalisation) (m_in::Bernoulli,) = Bernoulli(1 - mean(m_in))
