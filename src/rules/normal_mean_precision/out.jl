export rule

@rule NormalMeanPrecision(:out, Marginalisation) (m_μ::PointMass, m_τ::PointMass) = NormalMeanPrecision(mean(m_μ), mean(m_τ))

@rule NormalMeanPrecision(:out, Marginalisation) (q_μ::Any, q_τ::Any) = NormalMeanPrecision(mean(q_μ), mean(q_τ))

@rule NormalMeanPrecision(:out, Marginalisation) (m_μ::NormalMeanPrecision, q_τ::Any) = NormalMeanPrecision(mean(m_μ), cholinv( cov(m_μ) + cholinv(mean(q_τ)) ))