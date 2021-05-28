export rule

@rule MvNormalMeanPrecision(:out, Marginalisation) (m_μ::PointMass, m_Λ::PointMass) = MvNormalMeanPrecision(mean(m_μ), mean(m_Λ))

@rule MvNormalMeanPrecision(:out, Marginalisation) (q_μ::Any, q_Λ::Any) = MvNormalMeanPrecision(mean(q_μ), mean(q_Λ))

@rule MvNormalMeanPrecision(:out, Marginalisation) (m_μ::MvNormalMeanPrecision, q_Λ::Any) = MvNormalMeanPrecision(mean(m_μ), cholinv(cov(m_μ) + cholinv(mean(q_Λ))))

@rule MvNormalMeanPrecision(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, m_Λ::PointMass) = begin
    mμ, vμ = mean_cov(m_μ)
    return MvNormalMeanCovariance(mμ, vμ + cholinv(mean(m_Λ)))
end