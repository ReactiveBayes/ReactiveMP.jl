export GaussianMeanPrecision, GaussianMeanVariance
export MvGaussianMeanPrecision, MvGaussianMeanCovariance
export NormalDistributionsFamily, MvNormalDistributionsFamily

const GaussianMeanPrecision    = NormalMeanPrecision
const GaussianMeanVariance     = NormalMeanVariance
const MvGaussianMeanPrecision  = MvNormalMeanPrecision
const MvGaussianMeanCovariance = MvNormalMeanCovariance

const NormalDistributionsFamily   = Union{NormalMeanPrecision, NormalMeanVariance}
const MvNormalDistributionsFamily = Union{MvNormalMeanPrecision, MvNormalMeanCovariance}

import Base: prod

function Base.prod(::ProdPreserveParametrisation, left::NormalMeanVariance, right::NormalMeanPrecision)
    μ = (mean(left) * var(right) + mean(right) * var(left)) / (var(right) + var(left))
    v = (var(left) * var(right)) / (var(left) + var(right))
    return NormalMeanVariance(μ, v)
end

function Base.prod(::ProdPreserveParametrisation, left::NormalMeanPrecision, right::NormalMeanVariance) 
    p = precision(left) + precision(right)
    μ = (mean(left) * precision(left) + mean(right) * precision(right)) / p
    return NormalMeanPrecision(μ, p)
end

