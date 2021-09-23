## TODO: pointmass rules
## TODO: save sigma vectors in meta to limit allocations

@marginalrule Flow(:in) (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily, meta::FlowMeta{M,Linearization}) where { M } = begin
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
    (μ_in_hat, Ji) = backward_inv_jacobian(model, μ_out)
    Σ_in_hat = Ji * Σ_out * Ji'

    # calculate marginal distribution
    Λ_in     = cholinv(Σ_in) 
    Λ_in_hat = cholinv(Σ_in_hat)
    Λ_marg   = Λ_in + Λ_in_hat
    ξ_marg   = Λ_in * μ_in + Λ_in_hat * μ_in_hat

    # return marginal distribution
    return MvNormalWeightedMeanPrecision(ξ_marg, Λ_marg)

end


@marginalrule Flow(:in) (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily, meta::FlowMeta{M,Unscented}) where { M } = begin
    
    # extract parameters
    μ_out, Σ_out = mean_cov(m_out)

    # extract model
    model = getmodel(meta)
    T = eltype(model)

    # extract parameters of linearization
    approximation = getapproximation(meta)
    λ  = getλ(approximation)
    L  = getL(approximation)
    Wm = getWm(approximation)
    Wc = getWc(approximation)

    # calculate sigma points/vectors    
    sqrtΣ = sqrt((L + λ)*Σ_out)
    χ = Vector{Vector{Float64}}(undef, 2*L + 1)
    for k = 1:length(χ)
        χ[k] = copy(μ_out)
    end
    for l = 2:L+1
        χ[l]     .+= sqrtΣ[l-1,:]
        χ[L + l] .-= sqrtΣ[l-1,:]
    end

    # transform sigma points
    Y = backward.(model, χ)

    # calculate new parameters
    μ_in = zeros(T, L)
    Σ_in = zeros(T, L, L)
    for k = 1:2*L+1
        μ_in .+= Wm[k] .* Y[k]
    end
    for k = 1:2*L+1
        Σ_in .+= Wc[k] .* ( Y[k] - μ_in ) *  ( Y[k] - μ_in )'
    end

    # return distribution
    return prod(ProdAnalytical(), MvNormalMeanCovariance(μ_in, collect(Hermitian(Σ_in))), m_in)

end