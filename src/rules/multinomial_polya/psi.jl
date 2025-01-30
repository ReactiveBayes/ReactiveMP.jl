using PolyaGammaHybridSamplers

### When N = 1 , we will have one-hot-encoding for x representing it as categorical distribution
@rule MultinomialPolya(:ψ, Marginalisation) (q_x::PointMass, q_N::PointMass, m_ψ::GaussianDistributionsFamily, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    x = mean(q_x)
    N = mean(q_N)
    @assert sum(x) == N "The sum of the features must be equal to the total number of trials."

    T = promote_samplefloattype(q_x, q_N, m_ψ)
    
    K = length(x)
    x_cumsum = cumsum(x)
    prepended_cumsum = [0; x_cumsum[1:end-1]]  # Prepend zero for cumulative sums
    Nks = N .- prepended_cumsum[1:K-1]  # Corrected Nks calculation

    # Get current estimates from prior
    ψ_mean = mean(m_ψ)
    Λ_ψ = precision(m_ψ)
    η_ψ = Λ_ψ * ψ_mean

    if isnothing(meta)
        # Analytic version using expected values
        ω = Vector{T}(undef, K-1)
        for k in 1:K-1
            ω[k] = mean(PolyaGammaHybridSampler(Nks[k], ψ_mean[k]))
        end
    else
        # Monte Carlo version
        n_samples = getn_samples(meta)
        ψ_samples = rand(meta.rng, m_ψ, n_samples)
        ω_accum = zeros(T, K-1)
        
        for i in 1:n_samples
            ψ_i = ψ_samples[:, i]
            for k in 1:K-1
                sampler = PolyaGammaHybridSampler(Nks[k], ψ_i[k])
                ω_accum[k] += rand(meta.rng, sampler)
            end
        end
        ω = ω_accum ./ n_samples
    end

    η_new = η_ψ + (x[1:K-1] .- Nks ./ 2)
    Λ_new = Λ_ψ + Diagonal(ω)

    return MvGaussianWeightedMeanPrecision(η_new, Λ_new)
end
