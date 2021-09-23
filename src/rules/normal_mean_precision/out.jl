
# Belief Propagation                #
# --------------------------------- #
@rule NormalMeanPrecision(:out, Marginalisation) (m_μ::PointMass, m_τ::PointMass) = NormalMeanPrecision(mean(m_μ), mean(m_τ))

@rule NormalMeanPrecision(:out, Marginalisation) (m_μ::UnivariateNormalDistributionsFamily, m_τ::PointMass) = begin 
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return NormalMeanPrecision(m_μ_mean, inv(m_μ_cov + inv(mean(m_τ))))
end

# Variational                       # 
# --------------------------------- #
@rule NormalMeanPrecision(:out, Marginalisation) (q_μ::Any, q_τ::Any) = NormalMeanPrecision(mean(q_μ), mean(q_τ))

@rule NormalMeanPrecision(:out, Marginalisation) (m_μ::UnivariateNormalDistributionsFamily, q_τ::Any) = NormalMeanPrecision(mean(m_μ), cholinv( cov(m_μ) + cholinv(mean(q_τ)) ))