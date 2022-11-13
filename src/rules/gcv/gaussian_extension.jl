# This is an GCV extension for automatic rules transition with Gaussian Nodes

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::ExponentialLinearQuadratic, m_v::PointMass, meta::Any) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = NormalMeanVariance(m_out_mean, m_out_var), m_v = m_v, meta = meta)
end

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::ExponentialLinearQuadratic, q_v::Any, meta::Any) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = NormalMeanVariance(m_out_mean, m_out_var), q_v = q_v, meta = meta)
end

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::ExponentialLinearQuadratic, m_τ::PointMass, meta::Any) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_rule NormalMeanPrecision(:μ, Marginalisation) (m_out = NormalMeanVariance(m_out_mean, m_out_var), m_τ = m_τ, meta = meta)
end

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::ExponentialLinearQuadratic, q_τ::Any, meta::Any) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_rule NormalMeanPrecision(:μ, Marginalisation) (m_out = NormalMeanVariance(m_out_mean, m_out_var), q_τ = q_τ, meta = meta)
end

@marginalrule NormalMeanVariance(:out_μ_v) (m_out::ExponentialLinearQuadratic, m_μ::UnivariateNormalDistributionsFamily, m_v::PointMass, meta::Any) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_marginalrule NormalMeanVariance(:out_μ_v) (m_out = NormalMeanVariance(m_out_mean, m_out_var), m_μ = m_μ, m_v = m_v, meta = meta)
end

@marginalrule NormalMeanVariance(:out_μ) (m_out::ExponentialLinearQuadratic, m_μ::UnivariateNormalDistributionsFamily, q_v::Any, meta::Any) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_marginalrule NormalMeanVariance(:out_μ) (m_out = NormalMeanVariance(m_out_mean, m_out_var), m_μ = m_μ, q_v = q_v, meta = meta)
end

@marginalrule NormalMeanPrecision(:out_μ_τ) (m_out::ExponentialLinearQuadratic, m_μ::UnivariateNormalDistributionsFamily, m_τ::Any, meta::Any) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_marginalrule NormalMeanPrecision(:out_μ_τ) (m_out = NormalMeanVariance(m_out_mean, m_out_var), m_μ = m_μ, m_τ = m_τ, meta = meta)
end

@marginalrule NormalMeanPrecision(:out_μ) (m_out::ExponentialLinearQuadratic, m_μ::UnivariateNormalDistributionsFamily, q_τ::Any, meta::Any) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_marginalrule NormalMeanPrecision(:out_μ) (m_out = NormalMeanVariance(m_out_mean, m_out_var), m_μ = m_μ, q_τ = q_τ, meta = meta)
end
