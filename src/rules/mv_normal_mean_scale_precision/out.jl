# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScalePrecision(:out, Marginalisation) (q_μ::Any, q_γ::Any) = MvNormalMeanScalePrecision(
    mean(q_μ), mean(q_γ)
)

@rule MvNormalMeanScalePrecision(:out, Marginalisation) (
    m_μ::MultivariateNormalDistributionsFamily, q_γ::Any
) = begin
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return MvNormalMeanCovariance(
        m_μ_mean,
        m_μ_cov + inv(mean(q_γ)) * diageye(eltype(m_μ), ndims(m_μ))
    )
end

@rule MvNormalMeanScalePrecision(:out, Marginalisation) (
    m_μ::MvNormalMeanScalePrecision, q_γ::Any
) = begin
    m_out_mean = mean(m_μ)
    l_γ = m_μ.γ
    r_γ = mean(q_γ)
    return MvNormalMeanScalePrecision(m_out_mean, (l_γ*r_γ)/(l_γ+r_γ))
end
