export rule

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::PointMass, m_Σ::PointMass) = MvNormalMeanCovariance(mean(m_out), mean(m_Σ))
@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::MvNormalMeanCovariance, m_Σ::PointMass) = MvNormalMeanCovariance(mean(m_out), cov(m_out) + mean(m_Σ))

@rule MvNormalMeanCovariance(:μ, Marginalisation) (q_out::Any, q_Σ::Any) = MvNormalMeanCovariance(mean(q_out), mean(q_Σ))

# TODO check
@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::PointMass, q_Σ::Any) = MvNormalMeanCovariance(mean(m_out), mean(q_Σ))
@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::MvNormalMeanCovariance, q_Σ::Any) = MvNormalMeanCovariance(mean(m_out), cov(m_out) + mean(q_Σ))
