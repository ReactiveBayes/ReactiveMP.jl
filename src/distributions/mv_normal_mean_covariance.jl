export MvNormalMeanCovariance

import Distributions: mean, var, cov, std
import LinearAlgebra: diag
import Base: ndims

struct MvNormalMeanCovariance{T <: Real}
    mean       :: Vector{T}
    covariance :: PDMat{T,Array{T,2}}
end

Distributions.mean(nmc::MvNormalMeanCovariance) = nmc.mean
Distributions.var(nmc::MvNormalMeanCovariance)  = diag(cov(nmc))
Distributions.cov(nmc::MvNormalMeanCovariance)  = nmc.covariance
Distributions.std(nmc::MvNormalMeanCovariance)  = sqrt(Matrix(cov(nmc)))

Base.ndims(nmc::MvNormalMeanCovariance) = length(mean(nmc))

