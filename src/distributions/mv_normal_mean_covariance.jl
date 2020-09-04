export MvNormalMeanCovariance

import Distributions: mean, var, cov
import Base: ndims

struct MvNormalMeanCovariance{T}
    mean       :: Vector{T}
    covariance :: PDMat{T,Array{T,2}}
end

Distributions.mean(nmc::MvNormalMeanCovariance) = nmc.mean
Distributions.var(nmc::MvNormalMeanCovariance)  = diag(cov(nmc))
Distributions.cov(nmc::MvNormalMeanCovariance)  = nmc.covariance

Base.ndims(nmc::MvNormalMeanCovariance) = length(mean(nmc))

