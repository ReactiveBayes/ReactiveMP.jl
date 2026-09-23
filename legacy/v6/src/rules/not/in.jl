export rule

@rule NOT(:in, Marginalisation) (m_out::Bernoulli,) = Bernoulli(1 - mean(m_out))
