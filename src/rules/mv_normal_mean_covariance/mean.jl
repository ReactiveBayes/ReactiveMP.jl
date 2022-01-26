
# Belief Propagation                #
# --------------------------------- #
@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::PointMass, m_Σ::PointMass) = MvNormalMeanCovariance(mean(m_out), mean(m_Σ))

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_Σ::PointMass) = begin 
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + mean(m_Σ))
end

# Message with scale factor (BP case)  
@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::PointMass, m_Σ::PointMass, meta::ScaleFactorMeta) = begin
    message = @call_rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out = m_out, m_Σ = m_Σ) 
    scale = 0.0
    return ScaledMessage(message,scale)
end

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_Σ::PointMass, meta::ScaleFactorMeta) = begin
    message = @call_rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out = m_out, m_Σ = m_Σ) 
    scale = 0.0
    return ScaledMessage(message,scale)
end

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::ScaledMessage, m_Σ::PointMass, meta::ScaleFactorMeta) = begin 
    message = @call_rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out = m_out.message, m_Σ = m_Σ) 
    scale = m_out.scale
    return ScaledMessage(message, scale)
end

# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanCovariance(:μ, Marginalisation) (q_out::Any, q_Σ::Any) = MvNormalMeanCovariance(mean(q_out), mean(q_Σ))

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::PointMass, q_Σ::Any) = MvNormalMeanCovariance(mean(m_out), mean(q_Σ))

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_Σ::Any) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + mean(q_Σ))
end
