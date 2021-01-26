export rule

@rule Bernoulli(:p, Marginalisation) (m_out::PointMass{T}, ) where T = Beta(one(T) + mean(m_out), 2 * one(T) - mean(m_out))

@rule Bernoulli(:p, Marginalisation) (q_out::Any, ) = Beta(1.0 + mean(q_out), 2.0 - mean(q_out))