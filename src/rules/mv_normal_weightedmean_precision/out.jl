@rule MvNormalWeightedMeanPrecision(:out, Marginalisation) (m_ξ::PointMass, m_Λ::PointMass) = MvNormalWeightedMeanPrecision(mean(m_ξ), mean(m_Λ))

@rule MvNormalWeightedMeanPrecision(:out, Marginalisation) (q_ξ::Any, q_Λ::Any) = MvNormalWeightedMeanPrecision(mean(q_ξ), mean(q_Λ))
