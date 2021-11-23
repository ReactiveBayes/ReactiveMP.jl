## TODO: pointmass rules
## TODO: save sigma vectors in meta to limit allocations

@rule Flow(:in, Marginalisation) (m_out::MvNormalMeanCovariance, meta::FlowMeta{M,Linearization}) where { M } = begin
    
    # extract parameters
    μ_out, Σ_out = mean_cov(m_out)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    (μ_in, Ji) = backward_inv_jacobian(model, μ_out)
    Σ_in   = Ji * Σ_out * Ji'

    # return distribution
    return MvNormalMeanCovariance(μ_in, Σ_in)

end

@rule Flow(:in, Marginalisation) (m_out::MvNormalMeanPrecision, meta::FlowMeta{M,Linearization}) where { M } = begin
    
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

@rule Flow(:in, Marginalisation) (m_out::MvNormalWeightedMeanPrecision, meta::FlowMeta{M,Linearization}) where { M } = begin
    
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

@rule Flow(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, meta::FlowMeta{M,Unscented}) where { M } = begin
    
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
    χ = Vector{Vector{T}}(undef, 2*L + 1)
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
    return MvNormalMeanCovariance(μ_in, collect(Hermitian(Σ_in)))

end