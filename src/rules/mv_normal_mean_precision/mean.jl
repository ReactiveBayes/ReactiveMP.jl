export rule

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::PointMass, m_Λ::PointMass) = MvNormalMeanPrecision(mean(m_out), mean(m_Λ))

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_Λ::PointMass) = begin
    mout, vout = mean_cov(m_out)
    return MvNormalMeanCovariance(mout, vout + cholinv(mean(m_Λ)))
end

@rule MvNormalMeanPrecision(:μ, Marginalisation) (q_out::Any, q_Λ::Any) = MvNormalMeanPrecision(mean(q_out), mean(q_Λ))

@rule MvNormalMeanPrecision(:μ, Marginalisation) (m_out::MvNormalMeanPrecision, q_Λ::Any) = MvNormalMeanCovariance(mean(m_out), cov(m_out) + cholinv(mean(q_Λ)))
