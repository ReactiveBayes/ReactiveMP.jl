export NormalMeanPrecision

using Distributions

struct NormalMeanPrecision{T}
    mean      :: T
    precision :: T
end

NormalMeanPrecision(mean::Float64, precision::Float64) = NormalMeanPrecision{Float64}(mean, precision)

Distributions.mean(nmp::NormalMeanPrecision) = nmp.mean
Distributions.var(nmp::NormalMeanPrecision{T}) where T = one(T) / precision(nmp)

precision(nmp::NormalMeanPrecision) = nmp.precision

function Distributions.pdf(distribution:: NormalMeanPrecision{T}, x::T) where T
    return Distributions.pdf(Normal(mean(distribution), sqrt(var(distribution))), x)
end
