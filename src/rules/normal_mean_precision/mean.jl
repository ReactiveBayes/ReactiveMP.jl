export rule

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::PointMass, m_τ::PointMass) = NormalMeanPrecision(mean(m_out), mean(m_τ))

@rule NormalMeanPrecision(:μ, Marginalisation) (q_out::Any, q_τ::Any) = NormalMeanPrecision(mean(q_out), mean(q_τ))

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::NormalMeanPrecision, q_τ::Any) = NormalMeanPrecision(mean(m_out), cholinv( cov(m_out) + cholinv(mean(q_τ)) ))