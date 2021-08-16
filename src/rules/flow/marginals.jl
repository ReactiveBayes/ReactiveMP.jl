## TODO: pointmass rules

@marginalrule Flow(:in) (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily, meta::FlowMeta) = begin
    ## TODO: optimize for speed
    # Here, calculate q(out,μ,Σ) from μ(out)μ(μ)μ(Σ)f(out,μ,Σ). 
    # As m_out and m_Σ are pointmasses, we integrate these out and explicitly return them in a tuple.
    # Here q(μ) = q(μ | out, Σ)
    
    # extract parameters
    μ_in, Σ_in   = mean_cov(m_in)
    μ_out, Σ_out = mean_cov(m_out)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_in_hat = backward(model, μ_out)
    Ji       = inv_jacobian(model, μ_out)
    Σ_in_hat = Ji * Σ_out * Ji'

    # calculate marginal distribution
    Λ_in     = cholinv(Σ_in) 
    Λ_in_hat = cholinv(Σ_in_hat)
    Λ_marg   = Λ_in + Λ_in_hat
    ξ_marg   = Λ_in * μ_in + Λ_in_hat * μ_in_hat

    # return marginal distribution
    return MvNormalWeightedMeanPrecision(ξ_marg, Λ_marg)

end