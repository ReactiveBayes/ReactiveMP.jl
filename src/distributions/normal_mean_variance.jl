export NormalMeanVariance, MvNormalMeanCovariance

using Distributions
import PDMats: PDMat

struct NormalMeanVariance{T}
    mean     :: T
    variance :: T
end

Distributions.mean(nmp::NormalMeanVariance) = nmp.mean
Distributions.var(nmp::NormalMeanVariance)  = nmp.variance
Distributions.std(nmp::NormalMeanVariance)  = sqrt(var(nmp))
Distributions.cov(nmp::NormalMeanVariance)  = Distributions.var(nmp)

precision(nmp::NormalMeanVariance{T}) where T = one(T) / var(nmp)

function Distributions.pdf(distribution:: NormalMeanVariance, x)
    return Distributions.pdf(Normal(mean(distribution), std(distribution)), x)
end

struct MvNormalMeanCovariance{T}
    mean       :: Vector{T}
    covariance :: PDMat{T,Array{T,2}}
end

Distributions.mean(nmc::MvNormalMeanCovariance) = nmc.mean
Distributions.cov(nmc::MvNormalMeanCovariance)  = nmc.covariance

