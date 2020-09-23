export NormalMeanVariance

import Distributions: mean, var, std, cov, pdf
import PDMats: PDMat
import Base: precision

struct NormalMeanVariance{T <: Real}
    mean     :: T
    variance :: T
end

Distributions.mean(nmp::NormalMeanVariance) = nmp.mean
Distributions.var(nmp::NormalMeanVariance)  = nmp.variance
Distributions.std(nmp::NormalMeanVariance)  = sqrt(var(nmp))
Distributions.cov(nmp::NormalMeanVariance)  = Distributions.var(nmp)

Base.precision(nmp::NormalMeanVariance{T}) where T = one(T) / var(nmp)

function Distributions.pdf(distribution:: NormalMeanVariance, x)
    return Distributions.pdf(Normal(mean(distribution), std(distribution)), x)
end

