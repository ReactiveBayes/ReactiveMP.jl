using PolyaGammaHybridSamplers

@rule BinomialPolya(:β, Marginalisation) (
    q_y::Union{PointMass, Multinomial}, q_x::PointMass, q_n::PointMass, m_β::GaussianDistributionsFamily, meta::Union{BinomialPolyaMeta, Nothing}
) = begin
    y = mean(q_y)
    x = mean(q_x)
    n = mean(q_n)

    T = promote_samplefloattype(q_y, q_x, q_n, m_β)

    if isnothing(meta)
        β = mean(m_β)
        ωsampler = PolyaGammaHybridSampler(n, dot(x, β))
        ω_sample = convert(T, mean(ωsampler))
    else
        n_samples = getn_samples(meta)
        βsamples = rand(meta.rng, m_β, n_samples)
        ωsampler = map(βsample -> PolyaGammaHybridSampler(n, dot(x, βsample)), eachcol(βsamples))
        ω_samples = map(ωsampler -> rand(meta.rng, ωsampler), ωsampler)
        ω_sample = convert(T, mean(ω_samples))
    end

    κ = convert(T, y - n / 2)
    Λ = x * ω_sample * x'
    xi = κ * x

    return convert(promote_variate_type(typeof(xi), NormalWeightedMeanPrecision), xi, Λ)
end
