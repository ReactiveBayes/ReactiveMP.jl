export rule

@rule NOT(:in1, Marginalisation) (m_out::Bernoulli,) = Bernoulli(1 - mean(m_out))
