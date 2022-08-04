# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScalePrecision(:out, Marginalisation) (q_μ::Any, q_γ::Any) =
    MvNormalMeanPrecision(mean(q_μ), mean(q_γ) * diageye(ndims(q_μ)))

@rule MvNormalMeanScalePrecision(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, q_γ::Any) = begin
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return MvNormalMeanCovariance(m_μ_mean, m_μ_cov + cholinv(mean(q_γ) * diageye(ndims(m_μ))))
end
