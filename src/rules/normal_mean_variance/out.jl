export rule

@rule NormalMeanVariance(:out, Marginalisation) (m_μ::PointMass, m_v::PointMass) = NormalMeanVariance(mean(m_μ), mean(m_v))
@rule NormalMeanVariance(:out, Marginalisation) (m_μ::NormalMeanVariance, m_v::PointMass) = NormalMeanVariance(mean(m_μ), var(m_μ) + mean(m_v))

@rule NormalMeanVariance(:out, Marginalisation) (q_μ::Any, q_v::Any) = NormalMeanVariance(mean(q_μ), mean(q_v))