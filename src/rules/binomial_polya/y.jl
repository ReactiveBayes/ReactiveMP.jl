using PolyaGammaHybridSamplers
using ExponentialFamily.LogExpFunctions
@rule BinomialPolya(:y, Marginalisation) (q_x::PointMass, q_n::PointMass, q_β::GaussianDistributionsFamily, meta::Union{BinomialPolyaMeta, Nothing}) = begin
    x = mean(q_x)
    n = mean(q_n)
    if isnothing(meta)
        β = mean(q_β)
        ψ = dot(x, β)
        p = logistic(ψ)
        return Binomial(n, p)
    else
        n_samples = getn_samples(meta)
        βsamples = rand(meta.rng, q_β, n_samples)
        p_avg = mapreduce(βsample -> logistic(dot(x, βsample)), +, eachcol(βsamples))/n_samples
        return Binomial(n, p_avg)
    end
end