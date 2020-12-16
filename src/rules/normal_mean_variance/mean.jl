export rule

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::Dirac, m_v::Dirac) = NormalMeanVariance(mean(m_out), mean(m_v))
@rule NormalMeanVariance(:μ, Marginalisation) (m_out::NormalMeanVariance, m_v::Dirac) = NormalMeanVariance(mean(m_out), var(m_out) + mean(m_v))

@rule NormalMeanVariance(:μ, Marginalisation) (q_out::Any, q_v::Any) = NormalMeanVariance(mean(q_out), mean(q_v))