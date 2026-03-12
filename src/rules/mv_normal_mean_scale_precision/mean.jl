
# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (q_out::Any, q_γ::Any) = MvNormalMeanScalePrecision(mean(q_out), mean(q_γ))

@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_γ::Any) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + inv(mean(q_γ)) * diageye(samplefloattype(m_out), ndims(m_out)))
end

@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out::MvNormalMeanScalePrecision, q_γ::Any) = begin
    m_out_mean = mean(m_out)
    l_γ = m_out.γ
    r_γ = mean(q_γ)
    return MvNormalMeanScalePrecision(m_out_mean, (l_γ*r_γ)/(l_γ+r_γ))
end