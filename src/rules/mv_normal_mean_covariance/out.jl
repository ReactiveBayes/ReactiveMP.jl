
# Belief Propagation                #
# --------------------------------- #
@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::PointMass, m_Σ::PointMass) = MvNormalMeanCovariance(mean(m_μ), mean(m_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, m_Σ::PointMass) = begin 
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return MvNormalMeanCovariance(m_μ_mean, m_μ_cov + mean(m_Σ))
end

# Message with scale factor (BP case)
@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::PointMass, m_Σ::PointMass, meta::ScaleFactorMeta) = begin
    message = @call_rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ = m_μ, m_Σ = m_Σ) 
    scale = 0.0
    return ScaledMessage(message,scale)
end

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, m_Σ::PointMass, meta::ScaleFactorMeta) = begin
    message = @call_rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ = m_μ, m_Σ = m_Σ) 
    scale = 0.0
    return ScaledMessage(message,scale)
end

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::ScaledMessage, m_Σ::PointMass, meta::ScaleFactorMeta) = begin 
    message = @call_rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ = m_μ.message, m_Σ = m_Σ) 
    scale = m_μ.scale
    return ScaledMessage(message,scale)
end

# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanCovariance(:out, Marginalisation) (q_μ::Any, q_Σ::Any) = MvNormalMeanCovariance(mean(q_μ), mean(q_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::PointMass, q_Σ::Any) = MvNormalMeanCovariance(mean(m_μ), mean(q_Σ))

@rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, q_Σ::Any) = begin 
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return MvNormalMeanCovariance(m_μ_mean, m_μ_cov + mean(q_Σ))
end