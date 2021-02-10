export GammaShapeRate

import Distributions: Gamma, shape, rate
import SpecialFunctions: loggamma, digamma

struct GammaShapeRate{T <: Real}
    a :: T
    b :: T
end

GammaShapeRate(a::Real, b::Real)       = GammaShapeRate(promote(a, b)...)
GammaShapeRate(a::Integer, b::Integer) = GammaShapeRate(float(a), float(b))
GammaShapeRate()                       = GammaShapeRate(1.0, 1.0)

vague(::Type{ <: GammaShapeRate }) = GammaShapeRate(1.0, tiny)

function prod(::ProdPreserveParametrisation, left::GammaShapeRate{T}, right::GammaShapeRate{T}) where T
    return GammaShapeRate(shape(left) + shape(right) - one(T), rate(left) + rate(right))
end

Distributions.shape(dist::GammaShapeRate) = dist.a
Distributions.rate(dist::GammaShapeRate) = dist.b
Distributions.scale(dist::GammaShapeRate) = inv(dist.b)

Distributions.mean(dist::GammaShapeRate) = shape(dist)/ rate(dist)

Distributions.params(dist::GammaShapeRate) = (shape(dist), rate(dist))

function logmean(dist::GammaShapeRate)
    a, b = params(dist)
    return digamma(a) - log(b)
end


function Distributions.entropy(dist::GammaShapeRate)
    a, b = params(dist)
    return a - log(b) + loggamma(a) + (1-a)*digamma(a)
end
