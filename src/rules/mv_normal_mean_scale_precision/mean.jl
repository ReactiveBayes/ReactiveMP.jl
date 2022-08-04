
# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (q_out::Any, q_γ::Any) =
    MvNormalMeanPrecision(mean(q_out), mean(m_γ) * diageye(ndims(q_out)))

@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out::PointMass, q_γ::Any) =
    MvNormalMeanPrecision(mean(m_out), mean(m_γ) * diageye(ndims(q_out)))

@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_γ::Any) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + cholinv(mean(q_γ) * diageye(ndims(m_out))))
end
