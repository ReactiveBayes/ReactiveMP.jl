
# Belief Propagation                #
# --------------------------------- #
@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::PointMass, m_Σ::PointMass) = MvNormalMeanCovariance(mean(m_μ), mean(m_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, m_Σ::PointMass) = begin
    if isnothing(messages[1].addons)
        @logscale 0
    else 
        @logscale getlogscale(messages[1])
    end
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return MvNormalMeanCovariance(m_μ_mean, m_μ_cov + mean(m_Σ))
end

# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanCovariance(:out, Marginalisation) (q_μ::PointMass, q_Σ::PointMass) = MvNormalMeanCovariance(mean(q_μ), mean(q_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (q_μ::Any, q_Σ::Any) = MvNormalMeanCovariance(mean(q_μ), mean(q_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::PointMass, q_Σ::Any) = MvNormalMeanCovariance(mean(m_μ), mean(q_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, q_Σ::Any) = begin
    if isnothing(messages[1].addons)
        @logscale 0
    else 
        @logscale getlogscale(messages[1])
    end
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return MvNormalMeanCovariance(m_μ_mean, m_μ_cov + mean(q_Σ))
end
