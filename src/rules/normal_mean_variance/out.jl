
# Belief Propagation                #
# --------------------------------- #
@rule NormalMeanVariance(:out, Marginalisation) (m_μ::PointMass, m_v::PointMass) = NormalMeanVariance(mean(m_μ), mean(m_v))

@rule NormalMeanVariance(:out, Marginalisation) (m_μ::UnivariateNormalDistributionsFamily, m_v::PointMass) = begin
    @logscale 0
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return NormalMeanVariance(m_μ_mean, m_μ_cov + mean(m_v))
end

# Variational                       # 
# --------------------------------- #
@rule NormalMeanVariance(:out, Marginalisation) (q_μ::PointMass, q_v::PointMass) = NormalMeanVariance(mean(q_μ), mean(q_v))

@rule NormalMeanVariance(:out, Marginalisation) (q_μ::Any, q_v::Any) = NormalMeanVariance(mean(q_μ), mean(q_v))

@rule NormalMeanVariance(:out, Marginalisation) (m_μ::UnivariateNormalDistributionsFamily, q_v::Any) = begin
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return NormalMeanVariance(m_μ_mean, m_μ_cov + mean(q_v))
end

@rule NormalMeanVariance(:out, Marginalisation) (m_μ::UnivariateNormalDistributionsFamily, q_v::PointMass) = begin
    @logscale 0
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return NormalMeanVariance(m_μ_mean, m_μ_cov + mean(q_v))
end
