
# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (q_out::Any, q_γ::Any) =
    MvNormalMeanPrecision(mean(q_out), mean(q_γ) * diageye(samplefloattype(q_out), ndims(q_out)))

@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_γ::Any) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(
        m_out_mean,
        m_out_cov + inv(mean(q_γ)) * diageye(samplefloattype(m_out), ndims(m_out))
    )
end
