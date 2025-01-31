using PolyaGammaHybridSamplers

@rule MultinomialPolya(:ψ, Marginalisation) (q_x::PointMass, q_N::PointMass, m_ψ::GaussianDistributionsFamily, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    x = mean(q_x)
    N = mean(q_N)

    T = promote_samplefloattype(q_x, q_N, m_ψ)
    
    K = length(x)
    Nks = Vector{T}(undef, K-1)
    prev_sum = zero(T)
    @inbounds for k in 1:K-1
        Nks[k] = N - prev_sum
        if k < K-1
            prev_sum += x[k]
        end
    end
    if isnothing(meta)
        ω = map((n,x) -> mean(PolyaGammaHybridSampler(n,x)), Nks, mean(m_ψ))
    else
        n_samples = getn_samples(meta)
        ψ_samples = rand(meta.rng, m_ψ, n_samples)
        ω_accum = zeros(T, K-1)
        
        @inbounds for i in 1:n_samples
            @views ψ_i = ψ_samples[:, i]
            @inbounds for k in 1:K-1
                sampler = PolyaGammaHybridSampler(Nks[k], ψ_i[k])
                @views ω_accum[k] += rand(meta.rng, sampler)
            end
        end
        ω = ω_accum ./ n_samples
    end

    # Compute natural parameters for likelihood only
    Λ = Diagonal(ω)
    η = map((d,n) -> d - n/2, view(x, 1:K-1), Nks)

    # Return likelihood contribution without prior
    return MvGaussianWeightedMeanPrecision(η, Λ)
end
