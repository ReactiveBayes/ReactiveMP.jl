
# Belief Propagation                #
# --------------------------------- #
@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::PointMass, m_τ::PointMass) = NormalMeanPrecision(mean(m_out), mean(m_τ))

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_τ::PointMass) = begin
    @logscale 0
    m_out_mean, m_out_cov = mean_cov(m_out)
    return NormalMeanVariance(m_out_mean, m_out_cov + inv(mean(m_τ)))
end

# Variational                       # 
# --------------------------------- #
@rule NormalMeanPrecision(:μ, Marginalisation) (q_out::PointMass, q_τ::PointMass) = NormalMeanPrecision(mean(q_out), mean(q_τ))

@rule NormalMeanPrecision(:μ, Marginalisation) (q_out::Any, q_τ::Any) = NormalMeanPrecision(mean(q_out), mean(q_τ))

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, q_τ::Any) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    return NormalMeanVariance(m_out_mean, m_out_cov + cholinv(mean(q_τ)))
end

# GP meta                           # 
# --------------------------------- #
@rule NormalMeanPrecision(:μ, Marginalisation) (q_out::PointMass, q_τ::GammaShapeRate, meta::ProcessMeta) = begin 
    return @call_rule NormalMeanPrecision(:μ, Marginalisation) (q_out=q_out,q_τ=q_τ,meta=nothing)
end

@rule NormalMeanPrecision(:μ, Marginalisation) (q_out::PointMass, m_τ::GammaShapeRate, ) = begin 
    return @call_rule NormalMeanPrecision(:μ, Marginalisation) (q_out = q_out, q_τ = m_τ)
end