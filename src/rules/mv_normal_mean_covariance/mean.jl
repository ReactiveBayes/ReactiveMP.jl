
# Belief Propagation                #
# --------------------------------- #
@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::PointMass, m_Σ::PointMass) = MvNormalMeanCovariance(mean(m_out), mean(m_Σ))

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_Σ::PointMass) = begin
    @logscale 0
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + mean(m_Σ))
end

# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanCovariance(:μ, Marginalisation) (q_out::PointMass, q_Σ::PointMass) = MvNormalMeanCovariance(mean(q_out), mean(q_Σ))

@rule MvNormalMeanCovariance(:μ, Marginalisation) (q_out::Any, q_Σ::Any) = MvNormalMeanCovariance(mean(q_out), mean(q_Σ))

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::PointMass, q_Σ::Any) = MvNormalMeanCovariance(mean(m_out), mean(q_Σ))

@rule MvNormalMeanCovariance(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_Σ::Any) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out_mean, m_out_cov + mean(q_Σ))
end
