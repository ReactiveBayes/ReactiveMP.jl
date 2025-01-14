using PolyaGammaHybridSamplers

@rule BinomialPolya(:β, Marginalisation) (q_y::PointMass, q_x::PointMass, q_n::PointMass, m_β::GaussianDistributionsFamily, meta::Union{BinomialPolyaMeta, Nothing}) = begin
    y = mean(q_y)
    x = mean(q_x)
    n = mean(q_n)

    if isnothing(meta)
        β = mean(m_β)
        ωsampler = PolyaGammaHybridSampler(n, dot(x, β))
        ω_sample = mean(ωsampler)
    else
        n_samples = getn_samples(meta)
        βsamples = rand(m_β, n_samples)
        ωsampler = map(βsample -> PolyaGammaHybridSampler(n, dot(x, βsample)), eachcol(βsamples))
        ω_samples = map(ωsampler -> rand(ωsampler), ωsampler)
        ω_sample = mean(ω_samples)
    end

    κ = y - n / 2
    Λ = x * ω_sample * x'
    xi = κ * x

    return MvNormalWeightedMeanPrecision(xi, Λ)
end
