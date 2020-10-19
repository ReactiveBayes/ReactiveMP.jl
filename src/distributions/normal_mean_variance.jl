export NormalMeanVariance

import StatsFuns: log2π, invsqrt2π

struct NormalMeanVariance{T <: Real} <: ContinuousUnivariateDistribution
    μ :: T
    v :: T
end

NormalMeanVariance(μ::Real, v::Real)       = NormalMeanVariance(promote(μ, v)...)
NormalMeanVariance(μ::Integer, v::Integer) = NormalMeanVariance(float(μ), float(v))
NormalMeanVariance(μ::T) where {T <: Real} = NormalMeanVariance(μ, one(T))
NormalMeanVariance()                       = NormalMeanVariance(0.0, 1.0)

Distributions.mean(dist::NormalMeanVariance)      = dist.μ
Distributions.median(dist::NormalMeanVariance)    = dist.μ
Distributions.mode(dist::NormalMeanVariance)      = dist.μ
Distributions.var(dist::NormalMeanVariance)       = dist.v
Distributions.std(dist::NormalMeanVariance)       = sqrt(var(dist))
Distributions.cov(dist::NormalMeanVariance)       = var(dist)
Distributions.invcov(dist::NormalMeanVariance)    = inv(cov(dist))
Distributions.entropy(dist::NormalMeanVariance)   = (1 + log2π + log(var(dist))) / 2

Distributions.pdf(dist::NormalMeanVariance, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) / 2cov(dist))) / std(dist)
Distributions.logpdf(dist::NormalMeanVariance, x::Real) = -(log2π + log(var(dist)) + abs2(x - mean(dist)) / var(dist)) / 2

Base.precision(dist::NormalMeanVariance{T}) where T = invcov(dist)
Base.eltype(::NormalMeanVariance{T})        where T = T

Base.convert(::Type{ NormalMeanVariance{T} }, μ::S, v::S)                  where { T <: Real, S <: Real } = NormalMeanVariance(T(μ), T(v))
Base.convert(::Type{ NormalMeanVariance{T} }, dist::NormalMeanVariance{S}) where { T <: Real, S <: Real } = NormalMeanVariance(T(mean(dist)), T(var(dist)))

vague(::Type{ <: NormalMeanVariance }) = NormalMeanVariance(0.0, 1.0e20)

function Base.prod(::ProdPreserveParametrisation, left::NormalMeanVariance{T}, right::NormalMeanVariance{T}) where T 
    μ = (mean(left) * var(right) + mean(right) * var(left)) / (var(right) + var(left))
    v = (var(left) * var(right)) / (var(left) + var(right))
    return NormalMeanVariance{T}(μ, v)
end
