## TODO: pointmass rules
## TODO: save sigma vectors in meta to limit allocations

@rule Flow(:out, Marginalisation) (m_in::MvNormalMeanCovariance, meta::FlowMeta{M, <:Linearization}) where {M} = begin

    # extract parameters
    μ_in, Σ_in = mean_cov(m_in)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    (μ_out, J) = forward_jacobian(model, μ_in)
    Σ_out = J * Σ_in * J'

    # return distribution
    return MvNormalMeanCovariance(μ_out, Σ_out)
end

@rule Flow(:out, Marginalisation) (m_in::MvNormalMeanPrecision, meta::FlowMeta{M, <:Linearization}) where {M} = begin

    # extract parameters
    μ_in, Λ_in = mean_precision(m_in)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_out = forward(model, μ_in)
    Ji = inv_jacobian(model, μ_in)
    Λ_out = Ji' * Λ_in * Ji

    # return distribution
    return MvNormalMeanPrecision(μ_out, Λ_out)
end

@rule Flow(:out, Marginalisation) (m_in::MvNormalWeightedMeanPrecision, meta::FlowMeta{M, <:Linearization}) where {M} = begin

    # extract parameters
    μ_in, Λ_in = mean_precision(m_in)

    # extract model
    model = getmodel(meta)

    # calculate new parameters
    μ_out = forward(model, μ_in)
    Ji = inv_jacobian(model, μ_in)
    Λ_out = Ji' * Λ_in * Ji

    # return distribution
    return MvNormalMeanPrecision(μ_out, Λ_out)
end

@rule Flow(:out, Marginalisation) (m_in::MultivariateNormalDistributionsFamily, meta::FlowMeta{M, <:Unscented}) where {M} = begin

    # extract parameters
    μ_in, Σ_in = mean_cov(m_in)

    # extract model
    model = getmodel(meta)
    T = eltype(model)

    # extract parameters of linearization
    approximation = getapproximation(meta)
    λ = getλ(approximation)
    L = getL(approximation)
    Wm = getWm(approximation)
    Wc = getWc(approximation)

    # calculate sigma points/vectors    
    sqrtΣ = sqrt((L + λ) * Σ_in)
    χ = Vector{Vector{T}}(undef, 2 * L + 1)
    for k in 1:length(χ)
        χ[k] = copy(μ_in)
    end
    for l in 2:(L + 1)
        χ[l]     .+= sqrtΣ[l - 1, :]
        χ[L + l] .-= sqrtΣ[l - 1, :]
    end

    # transform sigma points
    Y = forward.(model, χ)

    # calculate new parameters
    μ_out = zeros(T, L)
    Σ_out = zeros(T, L, L)
    for k in 1:(2 * L + 1)
        μ_out .+= Wm[k] .* Y[k]
    end
    for k in 1:(2 * L + 1)
        Σ_out .+= Wc[k] .* (Y[k] - μ_out) * (Y[k] - μ_out)'
    end

    # return distribution
    return MvNormalMeanCovariance(μ_out, collect(Hermitian(Σ_out)))
end
