export rule

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::PointMass, m_Λ::PointMass) = MvNormalMeanPrecision(mean(m_out), mean(m_Λ))

@rule MvNormalMeanPrecision(:μ, Marginalisation) (q_out::Any, q_Λ::Any) = MvNormalMeanPrecision(mean(q_out), mean(q_Λ))

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::MvNormalMeanPrecision, q_Λ::Any) = MvNormalMeanPrecision(mean(m_out), cholinv(cov(m_out) + cholinv(mean(q_Λ))))
