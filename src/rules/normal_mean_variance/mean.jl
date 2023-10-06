
# Belief Propagation                #
# --------------------------------- #
@rule NormalMeanVariance(:μ, Marginalisation) (m_out::PointMass, m_v::PointMass) = NormalMeanVariance(mean(m_out), mean(m_v))

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_v::PointMass) = begin
    @logscale 0
    m_out_mean, m_out_cov = mean_cov(m_out)
    return NormalMeanVariance(m_out_mean, m_out_cov + mean(m_v))
end

# Variational                       # 
# --------------------------------- #
@rule NormalMeanVariance(:μ, Marginalisation) (q_out::PointMass, q_v::PointMass) = NormalMeanVariance(mean(q_out), mean(q_v))

@rule NormalMeanVariance(:μ, Marginalisation) (q_out::Any, q_v::Any) = NormalMeanVariance(mean(q_out), mean(q_v))

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::PointMass, q_v::Any) = NormalMeanVariance(mean(m_out), mean(q_v))

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, q_v::Any) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    return NormalMeanVariance(m_out_mean, m_out_cov + mean(q_v))
end
