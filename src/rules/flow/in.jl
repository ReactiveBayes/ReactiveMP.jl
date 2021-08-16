## TODO: pointmass rules

@rule Flow(:in, Marginalisation) (m_out::MvNormalMeanCovariance, meta::FlowMeta) = begin
    
    # extract parameters
    μ_out, Σ_out = mean_cov(m_out)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_in   = backward(model, μ_out)
    Ji     = inv_jacobian(model, μ_out)
    Σ_in   = Ji * Σ_out * Ji'

    # return distribution
    return MvNormalMeanCovariance(μ_in, Σ_in)

end

@rule Flow(:in, Marginalisation) (m_out::MvNormalMeanPrecision, meta::FlowMeta) = begin
    
    # extract parameters
    μ_out, Λ_out = mean_precision(m_out)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_in    = backward(model, μ_out)
    J       = jacobian(model, μ_out)
    Λ_in    = J' * Λ_out * J

    # return distribution
    return MvNormalMeanPrecision(μ_in, Λ_in)

end

@rule Flow(:in, Marginalisation) (m_out::MvNormalWeightedMeanPrecision, meta::FlowMeta) = begin
    
    # extract parameters
    μ_out, Λ_out = mean_precision(m_out)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_in    = backward(model, μ_out)
    J       = jacobian(model, μ_out)
    Λ_in    = J' * Λ_out * J

    # return distribution
    return MvNormalMeanPrecision(μ_in, Λ_in)

end