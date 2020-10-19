export NormalMeanPrecision

import StatsFuns: log2π, invsqrt2π

struct NormalMeanPrecision{T <: Real} <: ContinuousUnivariateDistribution
    μ :: T
    p :: T
end

NormalMeanPrecision(μ::Real, v::Real)       = NormalMeanPrecision(promote(μ, v)...)
NormalMeanPrecision(μ::Integer, v::Integer) = NormalMeanPrecision(float(μ), float(v))
NormalMeanPrecision(μ::T) where {T <: Real} = NormalMeanPrecision(μ, one(T))
NormalMeanPrecision()                       = NormalMeanPrecision(0.0, 1.0)

Distributions.mean(dist::NormalMeanPrecision)      = dist.μ
Distributions.median(dist::NormalMeanPrecision)    = dist.μ
Distributions.mode(dist::NormalMeanPrecision)      = dist.μ
Distributions.var(dist::NormalMeanPrecision)       = inv(dist.p)
Distributions.std(dist::NormalMeanPrecision)       = sqrt(var(dist))
Distributions.cov(dist::NormalMeanPrecision)       = var(dist)
Distributions.invcov(dist::NormalMeanPrecision)    = dist.p
Distributions.entropy(dist::NormalMeanPrecision)   = (1 + log2π - log(precision(dist))) / 2

Distributions.pdf(dist::NormalMeanPrecision, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) * precision(dist) / 2)) * sqrt(precision(dist))
Distributions.logpdf(dist::NormalMeanPrecision, x::Real) = -(log2π - log(precision(dist)) + abs2(x - mean(dist)) * precision(dist)) / 2

Base.precision(dist::NormalMeanPrecision{T}) where T = invcov(dist)
Base.eltype(::NormalMeanPrecision{T})        where T = T

Base.convert(::Type{ NormalMeanPrecision{T} }, μ::S, v::S)                   where { T <: Real, S <: Real } = NormalMeanPrecision(T(μ), T(v))
Base.convert(::Type{ NormalMeanPrecision{T} }, dist::NormalMeanPrecision{S}) where { T <: Real, S <: Real } = NormalMeanPrecision(T(mean(dist)), T(var(dist)))

vague(::Type{ <: NormalMeanPrecision }) = NormalMeanPrecision(0.0, 1.0e-20)

function Base.prod(::ProdPreserveParametrisation, left::NormalMeanPrecision{T}, right::NormalMeanPrecision{T}) where T 
    μ = (mean(left) * precision(left) + mean(right) * precision(right)) / (precision(left) + precision(right))
    p = precision(left) + precision(right)
    return NormalMeanPrecision{T}(μ, p)
end
