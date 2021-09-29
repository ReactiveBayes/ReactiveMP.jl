# This is an GCV extension for automatic rules transition with Gaussian Nodes

@rule NormalMeanVariance(:μ, Marginalisation) (m_out::ExponentialLinearQuadratic, q_v::Any) = begin 
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = NormalMeanVariance(m_out_mean, m_out_var), q_v = q_v)
end

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::ExponentialLinearQuadratic, q_τ::Any) = begin 
    m_out_mean, m_out_var = mean_var(m_out)
    return @call_rule NormalMeanPrecision(:μ, Marginalisation) (m_out = NormalMeanVariance(m_out_mean, m_out_var), q_τ = q_τ)
end