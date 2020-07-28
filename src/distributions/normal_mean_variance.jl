export NormalMeanVariance

using Distributions

struct NormalMeanVariance{T}
    mean     :: T
    variance :: T
end

Distributions.mean(nmp::NormalMeanVariance) = nmp.mean
Distributions.var(nmp::NormalMeanVariance)  = nmp.variance
Distributions.std(nmp::NormalMeanVariance)  = sqrt(var(nmp))

precision(nmp::NormalMeanVariance{T}) where T = one(T) / var(nmp)

function Distributions.pdf(distribution:: NormalMeanVariance, x)
    return Distributions.pdf(Normal(mean(distribution), std(distribution)), x)
end
