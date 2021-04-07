export Gamma, GammaShapeScale, GammaDistributionsFamily

import SpecialFunctions: digamma
import Distributions: Gamma, shape, scale
import StatsFuns: log2π

const GammaShapeScale             = Gamma
const GammaDistributionsFamily{T} = Union{GammaShapeScale{T}, GammaShapeRate{T}}

function logmean(dist::GammaShapeScale)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

function loggammamean(dist::GammaShapeScale)
    k, θ = params(dist)
    return 0.5 * (log2π - (digamma(k) + log(θ))) - mean(dist) * (1 + digamma(k + 1) + log(θ))
end

function meanlogmean(dist::GammaShapeScale)
    k, θ = params(dist)
    return mean(dist) * (digamma(k + 1) + log(θ))
end

vague(::Type{ <: GammaShapeScale }) = GammaShapeScale(1.0, huge)

function prod(::ProdPreserveParametrisation, left::GammaShapeScale, right::GammaShapeScale)
    return GammaShapeScale(shape(left) + shape(right) - 1.0, (scale(left) * scale(right)) / (scale(left) + scale(right)))
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

function prod(::ProdPreserveParametrisation, left::GammaShapeRate, right::GammaShapeScale)
    return GammaShapeRate(shape(left) + shape(right) - 1.0, rate(left) + rate(right))
end

function prod(::ProdPreserveParametrisation, left::GammaShapeScale, right::GammaShapeRate)
    # @show left
    # @show right
    return GammaShapeScale(shape(left) + shape(right) - 1.0, (scale(left) * scale(right)) / (scale(left) + scale(right)), check_args = true)
end
