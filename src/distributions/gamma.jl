export Gamma, GammaShapeScale, GammaDistributionsFamily

import SpecialFunctions: digamma
import Distributions: Gamma, shape, scale, cov
import StatsFuns: log2π

const GammaShapeScale             = Gamma
const GammaDistributionsFamily{T} = Union{GammaShapeScale{T}, GammaShapeRate{T}}

Distributions.cov(dist::GammaDistributionsFamily) = var(dist)

function mean(::typeof(log), dist::GammaShapeScale)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

function loggammamean(dist::GammaShapeScale)
    k, θ = params(dist)
    return 0.5 * (log2π - (digamma(k) + log(θ))) + mean(dist) * (-1 + digamma(k + 1) + log(θ))
end

function mean(::typeof(xtlog), dist::GammaShapeScale)
    k, θ = params(dist)
    return mean(dist) * (digamma(k + 1) + log(θ))
end

vague(::Type{ <: GammaShapeScale }) = GammaShapeScale(1.0, huge)

prod_analytical_rule(::Type{ <: GammaShapeScale }, ::Type{ <: GammaShapeScale }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::GammaShapeScale, right::GammaShapeScale)
    T = promote_type(eltype(left), eltype(right))
    return GammaShapeScale(shape(left) + shape(right) - one(T), (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

# Conversion to shape - scale parametrisation

function Base.convert(::Type{ GammaShapeScale{T} }, dist::GammaDistributionsFamily) where T
    return GammaShapeScale(convert(T, shape(dist)), convert(T, scale(dist)))
end

function Base.convert(::Type{ GammaShapeScale }, dist::GammaDistributionsFamily{T}) where T
    return convert(GammaShapeScale{T}, dist)
end

# Conversion to shape - rate parametrisation

function Base.convert(::Type{ GammaShapeRate{T} }, dist::GammaDistributionsFamily) where T
    return GammaShapeRate(convert(T, shape(dist)), convert(T, rate(dist)))
end

function Base.convert(::Type{ GammaShapeRate }, dist::GammaDistributionsFamily{T}) where T
    return convert(GammaShapeRate{T}, dist)
end

# Extensions of prod methods

prod_analytical_rule(::Type{ <: GammaShapeRate }, ::Type{ <: GammaShapeScale }) = ProdAnalyticalRuleAvailable()
prod_analytical_rule(::Type{ <: GammaShapeScale }, ::Type{ <: GammaShapeRate }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::GammaShapeRate, right::GammaShapeScale)
    T = promote_type(eltype(left), eltype(right))
    return GammaShapeRate(shape(left) + shape(right) - one(T), rate(left) + rate(right))
end

function prod(::ProdAnalytical, left::GammaShapeScale, right::GammaShapeRate)
    T = promote_type(eltype(left), eltype(right))
    return GammaShapeScale(shape(left) + shape(right) - one(T), (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

## Friendly functions

function logpdf_sample_friendly(dist::GammaDistributionsFamily)
    friendly = convert(GammaShapeScale, dist)
    return (friendly, friendly)
end