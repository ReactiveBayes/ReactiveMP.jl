using PolyaGammaHybridSamplers

@rule MultinomialPolya(:ψ, Marginalisation) (q_x::Any, q_N::PointMass, m_ψ::GaussianDistributionsFamily, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    x = mean(q_x)
    N = mean(q_N)
    T = promote_samplefloattype(q_x, q_N, m_ψ)
    K = length(x)
    Nks = compose_Nks(x, N)
    if isnothing(meta)
        ω = map((n, x) -> mean(PolyaGammaHybridSampler(n, x)), Nks, mean(m_ψ))
    else
        n_samples = getn_samples(meta)
        ψ_samples = rand(meta.rng, m_ψ, n_samples)
        ω_accum = zeros(T, K - 1)

        @inbounds for i in 1:n_samples
            @views ψ_i = ψ_samples[:, i]
            for k in 1:(K - 1)
                sampler = PolyaGammaHybridSampler(Nks[k], ψ_i[k])
                @views ω_accum[k] += rand(meta.rng, PolyaGammaHybridSampler(Nks[k], ψ_i[k]))
            end
        end
        ω = ω_accum ./ n_samples
    end

    
    η = map((d, n) -> d - n / 2, view(x, 1:(K - 1)), Nks)
    if length(η) == 1
        Λ = ω[1]
        η = η[1]
    else
        Λ = Diagonal(ω)
    end
    # Return likelihood contribution without prior
    # return  MvNormalWeightedMeanPrecision(η, Λ)
    dist = convert(promote_variate_type(typeof(η), NormalWeightedMeanPrecision), η, Λ)
    return dist
end

