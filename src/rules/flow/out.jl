## TODO: pointmass rules

@rule Flow(:out, Marginalisation) (m_in::MvNormalMeanCovariance, meta::FlowMeta) = begin
    
    # extract parameters
    μ_in, Σ_in = mean_cov(m_in)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_out   = forward(model, μ_in)
    J       = jacobian(model, μ_in)
    Σ_out   = J * Σ_in * J'

    # return distribution
    return MvNormalMeanCovariance(μ_out, Σ_out)

end

@rule Flow(:out, Marginalisation) (m_in::MvNormalMeanPrecision, meta::FlowMeta) = begin
    
    # extract parameters
    μ_in, Λ_in = mean_precision(m_in)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_out   = forward(model, μ_in)
    Ji      = inv_jacobian(model, μ_in)
    Λ_out   = Ji' * Λ_in * Ji

    # return distribution
    return MvNormalMeanPrecision(μ_out, Λ_out)

end

@rule Flow(:out, Marginalisation) (m_in::MvNormalWeightedMeanPrecision, meta::FlowMeta) = begin
    
    # extract parameters
    μ_in, Λ_in = mean_precision(m_in)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_out   = forward(model, μ_in)
    Ji      = inv_jacobian(model, μ_in)
    Λ_out   = Ji' * Λ_in * Ji

    # return distribution
    return MvNormalMeanPrecision(μ_out, Λ_out)

end