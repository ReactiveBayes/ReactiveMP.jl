using PolyaGammaHybridSamplers

@rule MultinomialPolya(:ψ, Marginalisation) (q_x::Any, q_N::Union{PointMass, Poisson, Binomial, Categorical}, m_ψ::GaussianDistributionsFamily, meta::MultinomialPolyaMeta) = begin
    x = mean(q_x)
    N = mode(q_N)
    T = promote_samplefloattype(q_x, q_N, m_ψ)
    K = length(x)
    Nks = compose_Nks(x, N)

    ω = map((n, x) -> mean(PolyaGammaHybridSampler(n, x)), Nks, mean(m_ψ))

    η = map((d, n) -> d - n / 2, view(x, 1:(K - 1)), Nks)
    if length(η) == 1
        Λ = ω[1]
        η = η[1]
    else
        Λ = Diagonal(ω)
    end

    return convert(promote_variate_type(typeof(η), NormalWeightedMeanPrecision), η, Λ)
end
