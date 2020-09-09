export NormalMeanPrecision

import Distributions: mean, var, std, pdf
import Base: precision

struct NormalMeanPrecision{T <: Real}
    mean      :: T
    precision :: T
end

Distributions.mean(nmp::NormalMeanPrecision)           = nmp.mean
Distributions.var(nmp::NormalMeanPrecision{T}) where T = one(T) / precision(nmp)
Distributions.std(nmp::NormalMeanPrecision)            = sqrt(var(nmp))

Base.precision(nmp::NormalMeanPrecision) = nmp.precision

function Distributions.pdf(distribution::NormalMeanPrecision, x)
    return Distributions.pdf(Normal(mean(distribution), std(distribution)), x)
end
