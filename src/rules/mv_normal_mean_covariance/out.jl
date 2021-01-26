export rule

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::PointMass, m_Σ::PointMass) = MvNormalMeanCovariance(mean(m_μ), mean(m_Σ))
@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::MvNormalMeanCovariance, m_Σ::PointMass) = MvNormalMeanCovariance(mean(m_μ), cov(m_μ) + mean(m_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (q_μ::Any, q_Σ::Any) = MvNormalMeanCovariance(mean(q_μ), mean(q_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::MvNormalMeanCovariance, q_Σ::Any) = MvNormalMeanCovariance(mean(m_μ), cov(m_μ) + mean(q_Σ))